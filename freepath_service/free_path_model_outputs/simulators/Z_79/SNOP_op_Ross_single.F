#define cubareal real*REALSIZE
	program opa_SNOP_single
		use par_register
		use functions
		use cubapar
		use gpu_bridge
		implicit none

	integer ndim, ncomp, nvec, last, seed, mineval, maxeval
	cubareal epsrel, epsabs, userdata
	parameter (ndim = 1)
	parameter (ncomp = 1)
	parameter (userdata = 0)
	parameter (nvec = 1000000)
	parameter (epsrel = 1D-8)
	parameter (epsabs = 1D-12)
	parameter (last = 4)
	parameter (seed = 0)
	parameter (mineval = 0)
	parameter (maxeval = 30000000)

	integer nstart, nincrease, nbatch, gridno
	integer*8 spin
	character*(*) statefile
	parameter (nstart = 10000000)
	parameter (nincrease = 5000000)
	parameter (nbatch = 2000000)
	parameter (gridno = 0)
	parameter (statefile = "")
	parameter (spin = -1)

	external integrand

	cubareal integral(ncomp), error(ncomp), prob(ncomp)
	integer verbose, neval, fail
	character*128 arg
	character*16 env
	integer c, gpu_status, argc
	integer :: time0, time1, clock_rate
	real(8) :: elapsed_s, log_lnu

	call system_clock(time0, clock_rate)

	argc = command_argument_count()
	if (argc .lt. 3) then
	   print *, 'SNOP_SINGLE_ERROR expected arguments: rod tep tgama'
	   stop 2
	end if
	call get_command_argument(1, arg)
	read(arg, *) rod
	call get_command_argument(2, arg)
	read(arg, *) tep
	call get_command_argument(3, arg)
	read(arg, *) tgama

	call getenv("CUBAVERBOSE", env)
	verbose = 2
	read(env, *, iostat=fail, end=999, err=999) verbose
999	continue

	dfree=10.d0
	LTE_fac=1.d0
	tol=1.d-6
	fffac=1.d0
	bffac=1.d0
	bbfac=1.d0
	esfac=1.d0

	sig=0.d0
	open(10, file='screen_constants.txt', status='old')
	do i=1, n
	   read(10, *, iostat=ios) (sig(j, i), j=1, n)
	end do
	close(10)

	Fnm=0.d0
	open(20, file='oscillator_strengths.txt', status='old')
	do i=1, n-1
	   read(20, *, iostat=ios) (Fnm(j, i+1), j=1, i)
	end do
	close(20)

	fnmin=0.d0
	do i=1, n-1
	   do j=i+1, n
	      if(i .gt. 4) then
	         fnmin(i, j)=1.96d0*i*j**3/(j**2-i**2)**3
	      else
	         if(j .gt. 5) then
	            fnmin(i, j)=1.96d0*i*j**3/(j**2-i**2)**3
	         else
	            fnmin(i, j)=Fnm(i, j)
	         end if
	      end if
	   end do
	end do

	Ni=rod*NA/Aw

	call Enoutpre_pro2(Zan, n, tep, rod, Aw, sig,
     &             Qnout, Enout_pre)

	call sbar_pro_More(Zan, tep, rod, Aw, sbar_More)
	ne_in=sbar_More*Ni

	do itern=1, max_iter
	   call SNOP_nsp2(Zan, n, tep, rod, Aw, Ni, dfree,
     &                LTE_fac, ne_in, Enout_pre, ne_out, Enout_out,
     &                ssbrnsp)

	   fne=ne_in-ne_out
	   fnep=fne/ne_in
	   norm_fnep=abs(fnep)

	   if (norm_fnep > 1.0d5) then
	      print *, 'SNOP_SINGLE_ERROR divergent Newton iteration'
	      stop 3
	   end if

	   if(norm_fnep .le. tol) then
	      exit
	   end if

	   call SNOP_nsp_jac2(Zan, n, tep, rod, Aw, Ni, dfree,
     &                    LTE_fac, ne_in, Enout_pre, jac)
	   dne=-fne
	   call dgesv(1, nrhs, jac, 1, ipiv, dne, 1, info)
	   if(info .ne. 0) then
	      print *, 'SNOP_SINGLE_ERROR DGESV failed'
	      stop 4
	   end if
	   ne_in=ne_in+dne
	end do

	gpu_status = init_gpu_bridge(Zan, n, tep, rod, Aw, Ni, sig,
     &       fnmin, ssbrnsp, Qnout, Enout_out, ne_out, tgama, lbfac,
     &       fffac, bffac, bbfac, esfac)
	call setup_cuba_parallel(gpu_status)

	call vegas(ndim, ncomp, integrand, userdata, nvec,
     &    epsrel, epsabs, verbose, seed,
     &    mineval, maxeval, nstart, nincrease, nbatch,
     &    gridno, statefile, spin,
     &    neval, fail, integral, error, prob)

	lnu=1.d0/rod*integral(1)
	log_lnu=dlog10(lnu)

	call system_clock(time1, clock_rate)
	elapsed_s=dble(time1-time0)/dble(clock_rate)

	write(*,*) 'SNOP_SINGLE_RESULT', Zan, rod, tep, tgama,
     &       lnu, log_lnu, elapsed_s, neval, fail

	end

	integer function integrand(ndim, xx, ncomp, ff, userdata,
     &          nvec, core, weight, iter)
	use par_register
	use functions
	use cubapar
	use gpu_bridge
	implicit none

	integer ndim, ncomp, nvec, core, iter
	integer ivec, gpu_ok
	cubareal xx(*), ff(*)
	cubareal userdata, weight(*)

	if (core .lt. 0) then
	   gpu_ok = eval_gpu_bridge(nvec, xx, ff, core)
	   if (gpu_ok .eq. 1) then
	      integrand = 0
	      return
	   end if
	end if

	do ivec=1, nvec
	   ue=xx(ivec)/(1.d0-xx(ivec))
	   slo_ue=1.d0/(1.d0-xx(ivec))**2
	   hplnu=ue*tep

	   call abs_coe4(Zan, n, tep, rod, Aw, Ni, sig, fnmin,
     &       ssbrnsp, Qnout, Enout_out, ne_out, hplnu, lbfac, fffac,
     &       bffac, bbfac, esfac, emitot, abstot)

	   gru2=15.d0/(4.d0*pi**4)*(tep/tgama)**5*ue**4*
     &       Dexp((tep/tgama)*ue)/(dexp((tep/tgama)
     &       *ue)-1.d0)**2

	   ff(ivec)=1.d0/abstot*gru2*slo_ue
	end do

	integrand = 0
	return
	end

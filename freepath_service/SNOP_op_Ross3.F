#define cubareal real*REALSIZE
	program opa_SNOP
		use par_register  !包含计算所公用的参数
		use functions  !包含计算所需函数
		use cubapar  !加速计算
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
	integer verbose, nregions, neval, fail
	character*16 env
	
		integer c
		integer gpu_status
	
	!program running time
	integer :: time0, time1, run_time

	call system_clock(time0)
	
	!cuba code
	call getenv("CUBAVERBOSE", env)
	verbose = 2
	read(env, *, iostat=fail, end=999, err=999) verbose
999	continue
	
	!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@	
	!这些参数不必修改
	dfree=10.d0  !双电子复合参数
	LTE_fac=1.d0 !控制alphas1是否为0，LTE_fac=1，alphas1不变；LTE_fac=0，alphas1=0	
	tol=1.d-6 !(ne_in-ne_out)/ne的绝对值应小于此数
	
	fffac=1.d0 !ff过程贡献输出控制参数，fffac=1，总贡献中保留ff过程贡献；fffac=0，总贡献中不包括ff过程的贡献
	bffac=1.d0 !同上
	bbfac=1.d0 !同上
	esfac=1.d0 !同上
	!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	
	sig=0.d0  !先把所有元素都赋0
	open(10, file='screen_constants.txt', status='old')
	do i=1, n
	   read(10, *, iostat=ios) (sig(j, i), j=1, n)
	   do j=1, n
	   print*, 'sig(', j, ',', i, ')=', sig(j, i) 
           end do
	end do

	close(10)
	
	Fnm=0.d0  !先把所有元素都赋0
	open(20, file='oscillator_strengths.txt', status='old')
	do i=1, n-1
	   read(20, *, iostat=ios) (Fnm(j, i+1), j=1, i)
	   do j=1, i
	   print*, 'Fnm(', j, ',', i+1, ')=', Fnm(j, i+1)
	   end do 
	end do 
	
	close(20)
	
	fnmin=0.d0  !先把所有元素都赋0
	
	do i=1, n-1
	   do j=i+1, n
	        if(i .gt. 4) then
	           fnmin(i, j)=1.96d0*i*j**3/(j**2-i**2)**3
	        else
	           if(j .gt. 5) then
	           fnmin(i, j)=1.96d0*i*j**3/(j**2-i**2)**3
	           else
	           fnmin(i, j)=Fnm(i, j)  !精确值
	           end if
	         end if
	   end do
	end do
	
	!相对原子质量（取自多种元素的核外电子排布）：
	!Be 9.01, C 12.01, Al 26.98, Ti 47.87, Fe 55.85, Cu 63.55, Mo 95.94
	!Sn 118.71, Ba 137.3, Eu 151.96, W 183.84, Au 196.97, Pb 207.2, U 238.0
	
	Aw=196.97d0  !相对原子质量
	lbfac=1000.d0 !公式(28)的 w 的放大系数
	
	calnum=1
	
	do i2=1, ndir !rod取不同值
	rod=(rod_high-rod_low)/ndir*i2+rod_low
	Ni=rod*NA/Aw  !Ni由密度、阿伏伽德罗常数和相对原子质量求出
	
	do j2=1, ndie !tep取不同值
	tep=(tep_high-tep_low)/ndie*j2+tep_low
	
	do k2=1, ndig !tgama取不同值
	tgama=(tgama_high-tgama_low)/ndig*k2+tgama_low
	
	print*, '计算到第', calnum, '组数据'
	
	!计算能级中通用的部分
	call Enoutpre_pro2(Zan, n, tep, rod, Aw, sig, 
     &             Qnout, Enout_pre)
	
	!采用More公式计算粗略的电离度 sbar_More	
	call sbar_pro_More(Zan, tep, rod, Aw, sbar_More)
	
	ne_in=sbar_More*Ni
	
	do itern=1, max_iter  !最多迭代max_iter次
	
	call SNOP_nsp2(Zan, n, tep, rod, Aw, Ni, dfree, 
     &                LTE_fac, ne_in, Enout_pre, ne_out, Enout_out,
     &                ssbrnsp)
	
	fne=ne_in-ne_out
	fnep=fne/ne_in
	
	norm_fnep=abs(fnep)
	
	if (norm_fnep > 1.0d5) then
            print*, '警告: 函数值过大，可能发散!'
            stop  !停止计算
	end if
	
	!检查收敛性	
	if(norm_fnep .le. tol) then
	exit  !停止计算，跳出该循环
	   
	end if
	
	call SNOP_nsp_jac2(Zan, n, tep, rod, Aw, Ni, dfree, 
     &                    LTE_fac, ne_in, Enout_pre, jac)   
	
	!牛顿法新的nsp_{i+1}与原本的nsp_i的关系为nsp_{i+1}=nsp_{i}-jac^{-1}(x_i)*fnsp(x_i)，引入jac(x_i)*dnsp=-fnsp(x_i)，接下来需要求解此线性方程组，从而得到dnsp的结果
	   !使用LAPACK中的DGESV子程序来求解线性方程组 jac * dnsp = -fnsp
	   !注意：DGESV会改变输入的矩阵jac和向量dnsp（dnsp在调用时是-fnsp，返回时是解向量）。   
	dne=-fne
	
	call dgesv(1, nrhs, jac, 1, ipiv, dne, 1, info)
	   
	if(info .ne. 0) then
	      print*, '出错'
	      stop
	end if   
     
	ne_in=ne_in+dne
	
	if (itern == max_iter) then
            print*, '警告: 达到最大迭代次数!'
        end if
           
	end do
	 
	!@@@@@@@@@@@@@@@@@@@@ 以上为计算sbar和nsp(s)的部分 @@@@@@@@@@@@@@@@@@@ 
	
		gpu_status = init_gpu_bridge(Zan, n, tep, rod, Aw, Ni, sig,
     &       fnmin, ssbrnsp, Qnout, Enout_out, ne_out, tgama, lbfac,
     &       fffac, bffac, bbfac, esfac)
		call setup_cuba_parallel(gpu_status)
	
	call vegas(ndim, ncomp, integrand, userdata, nvec,
     &    epsrel, epsabs, verbose, seed,
     &    mineval, maxeval, nstart, nincrease, nbatch,
     &    gridno, statefile, spin,
     &    neval, fail, integral, error, prob)
     
	print *, "neval    =", neval
	print *, "fail     =", fail
	print '(F20.12," +- ",F20.12,"   p = ",F8.3)',
     &    (integral(c), error(c), prob(c), c = 1, ncomp)
     
	lnu=1.d0/rod*integral(1)
	!print*, 'free path is', lnu
	
	open(unit=21, file='data.txt', status='unknown', 
     &       action='write', position='append')
	write(21, '(4F20.12)') rod, tep, tgama, lnu
	flush(21)
	close(21)
		
	calnum=calnum+1
	
	end do
	end do
	end do	
	
	call system_clock(time1)
	run_time = time1-time0
	print* , 'program running time', run_time*0.001d0, 's'

	end
	
	integer function integrand(ndim, xx, ncomp, ff, userdata,
     &          nvec, core, weight, iter)
	use par_register  !包含计算所公用的参数
	use functions  !包含计算所需函数
	use cubapar  !加速计算
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
		!这里按照之前的写法仍用 u 作为变量，实际上 u 就是SNOP里面的 x
		!由于蒙卡积分变量取值范围是(0,1)，因此需要做变量替换
		!u=x/(1-x), x取值(0,1), 则有du=dx/(1-x)^2
		
	ue=xx(ivec)/(1.d0-xx(ivec))
	
	slo_ue= 1.d0/(1.d0-xx(ivec))**2 !u->x的jac
	
	!hplnu=hpl*nu  
	hplnu=ue*tep
	
	call abs_coe4(Zan, n, tep, rod, Aw, Ni, sig, fnmin, 
     &       ssbrnsp, Qnout, Enout_out, ne_out, hplnu, lbfac, fffac, 
     &       bffac, bbfac, esfac, emitot, abstot)
     

	gru2=15.d0/(4.d0*pi**4)*(tep/tgama)**5*ue**4*
     &       Dexp((tep/tgama)*ue)/(dexp((tep/tgama)
     &       *ue)-1.d0)**2 !3-temperature weighting function G_R(u)
	
	ff(ivec)=1.d0/abstot*gru2*slo_ue  !inverse Rosseland opacity 
	end do
	
	integrand = 0
	return 
	end

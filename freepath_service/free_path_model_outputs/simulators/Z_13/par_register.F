	module par_register
	implicit none
	save

	integer, parameter :: Zan=13
	integer, parameter :: n=3

	real(8) :: rod, tep, tgama, lnu
	real(8) :: Aw=26.98d0, Ni
	real(8) :: NA=6.022d23
	real(8) :: sig(n,n), Fnm(n-1,n), fnmin(n-1,n)
	real(8), parameter :: pi=4.d0*DATAN(1.d0)
	real(8) :: dfree
	integer :: i, j, itern, max_iter=100
	real(8) :: tol

	real(8) :: Qnout(0:Zan), Enout_pre(0:Zan)
	real(8) :: ssbrnsp(0:Zan)
	real(8) :: sbar_More, LTE_fac
	real(8) :: ne_in, ne_out
	real(8) :: Enout_out(0:Zan)
	real(8) :: fne, fnep, norm_fnep
	real(8) :: jac, dne

	real(8) :: hplnu, lbfac=300d0
	real(8) :: fffac, bffac, bbfac, esfac
	real(8) :: emitot, abstot

	integer, parameter :: nrhs=1
	integer :: ipiv(1)
	integer :: info
	integer :: ios

	real(8) :: ue, slo_ue
	real(8) :: gru2

	end module

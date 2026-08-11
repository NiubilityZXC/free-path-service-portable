	module par_register
	implicit none
	save
	
	!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	!只用修改此处
	!元素        原子序数    核外电子排布
	!铍（Be）：     4		 2层，2-2
	!碳（C）	       6	2层，2-4
	!铝（Al）：     13       3层，2-8-3
	!钛（Ti）：     22      4层，2-8-10-2
	!铁（Fe）：     26      4层，2-8-14-2                      
	!铜（Cu）：     29      4层，2-8-18-1  
	!钼（Mo）：     42      5层，2-8-18-13-1
	!锡（Sn）：     50      5层，2-8-18-18-4
	!钡（Ba）：     56     6层，2-8-18-18-8-2
	!铕（Eu）：     63     6层，2-8-18-25-8-2
	!钨（W）：      74     6层，2-8-18-32-12-2
	!金（Au）：     79     6层，2-8-18-32-18-1
	!铅（Pb）：     82     6层，2-8-18-32-18-4
	!铀（U）：      92    7层，2-8-18-32-21-9-2
	integer, parameter :: Zan=79 !原子序数
	integer, parameter :: n=6  !原子核外电子层数
	integer, parameter :: ndir=50, ndie=50, ndig=50 !等离子体密度，电子温度，光子温度取值的个数
	real(8) :: rod_low=1.d-4, rod_high=3.d4  !等离子体密度的上下限
	real(8) :: tep_low=0.03d0, tep_high=2.d5  !电子温度的上下限
	real(8) :: tgama_low=0.01d0, tgama_high=3.d4  !光子温度的上下限
	!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	
	real(8) :: rod, tep, tgama, lnu !等离子体密度，电子温度，光子温度，由不透明度算出的自由程
	real(8) :: Aw, Ni  !等离子体相对原子质量, 数密度
	real(8) :: NA=6.022d23  !阿伏伽德罗常数
	real(8) :: sig(n,n), Fnm(n-1,n), fnmin(n-1,n)  !屏蔽系数，hydrogen oscillator strength fnm，输入程序的fnm
	real(8), parameter :: pi=4.d0*DATAN(1.d0) !圆周率 pi
	real(8) :: dfree !双电子复合参数
	integer :: i, j, itern, max_iter=100
	real(8) :: tol !(ne_in-ne_out)/ne的绝对值应小于此数

	real(8) :: Qnout(0:Zan), Enout_pre(0:Zan)  !不同电离度情况下最外层的有效电荷, 能级中只跟电离度有关的部分
	real(8) :: ssbrnsp(0:Zan) !符合SNOP公式的离子数密度概率
	!integer :: first_idx, last_idx
	real(8) :: sbar_More, LTE_fac !由More公式计算的平均电离度，控制alphas1是否为0的参数
	real(8) :: ne_in, ne_out !自由电子数密度初值，由初值算出的次值
	real(8) :: Enout_out(0:Zan) !电离能
	real(8) :: fne, fnep, norm_fnep !ne_in-ne_out，fne/ne_in，|fnep|
	real(8) :: jac, dne !ne的导数矩阵，-fne
	
	real(8) :: hplnu, lbfac !光子能量，wlb放大系数
	real(8) :: fffac, bffac, bbfac, esfac !控制emitot和abstot里面显示哪些项，例如fffac=1表示有ff过程的贡献，0则表示没有，这样可以单独得到某个过程的贡献
	real(8) :: emitot, abstot !3种过程总的发射系数，总的吸收系数
	
	!********  使用Lapack会用到的参数  ************
	integer, parameter :: nrhs=1 !等号右边项个数
	integer :: ipiv(1) !记录置换信息
	integer :: info !lapack输出，0表示成功
	!********************************************
	
	integer :: ios  !如果 open 语句成功，则返回的 ios 值为零，否则为非零值。
	
	real(8) :: ue, slo_ue  !hpl*nu/tep，u变换到x1需要乘的jac系数
	real(8) :: gru2 !weighting function G_R(u)
	
	integer :: i2, j2, k2, calnum
	
	end module 
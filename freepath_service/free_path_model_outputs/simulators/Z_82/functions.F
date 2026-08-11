	module functions
	contains
	
	subroutine SNOP_nsp2(Zan, n, tep, rod, Aw, Ni, dfree, 
     &                      LTE_fac, ne_in, Enout_pre, ne_out, 
     &                      Enout_out, ssbrnsp)
	implicit none
	
	!************  通用变量  *********************
	integer, parameter :: max_principal = 7  ! 最大主量子数
	integer, intent(in) :: Zan, n  !元素的原子序数，束缚能层数
	real(8) :: tep, rod, Aw, Ni  !等离子体温度, 密度, 相对原子质量, 数密度
	real(8) :: dfree  !(11)式中的自由参数 d
	real(8) :: LTE_fac !控制alphas1是否为0，LTE_fac=1，alphas1不变；LTE_fac=0，alphas1=0
	real(8) :: Enout_pre(0:Zan) !Enout的前半部分，跟Qnout有关，是通用的
	!********************************************
	
	!*********  此子程序所需变量  *****************
	real(8) :: ne_in !输入的自由电子数密度
	real(8) :: sbar_in, del_En_in, Enout_in(0:Zan) !由ne_in算出的平均电离度，delta En，Enout
	integer :: i, j
	integer :: chrem, pn_ns(max_principal), nout  !束缚电子总数，s度电离情况下各层束缚电子数，束缚电子占据的最外层主量子数
	integer :: nouts1, pn_nss1  !s+1度电离时s电离的最外层主量子数，s+1度电离时s电离最外层电子数
	real(8) :: chis, del_s, ksis1  !(12)-(14)式用到的 \chi_s，\delta_s，\ksi_{s+1}
	real(8) :: Cs, betas1, alphas1  !(12)-(14)式等号左边的3个变量
	real(8) :: ssbrnsp(0:Zan)  !由ne_in算出的不同电离度离子数密度概率
	real(8) :: ssbrnspsum !ssbrnsp相加之和
	real(8) :: sbar_out, ne_out, del_En_out, Enout_out(0:Zan)  !由ssbrnsp算出的sbar，ne，delta En，Enout
	real(8) :: fac(0:Zan-1), term, term_sum !ssbrnsp(s+1)/ssbrnsp(s)的系数，fac乘积，fac乘积相加之和
	!********************************************
	
	sbar_in=ne_in/Ni
	
	del_En_in=36.d0*sbar_in*(rod/Aw)**(1.d0/3.d0)  !单位 eV
	
	Enout_in=Enout_pre-del_En_in !最外层的能级，单位 eV
	
	term_sum=0.d0
	term=1.d0
	ssbrnsp=0.d0
	fac=0.d0
	
	do i=0, Zan-1
	chrem=Zan-i !剩余电荷
	
	call pn_ns_Z2(Zan, n, chrem, pn_ns, nout, nouts1, 
     &               pn_nss1)
		
	chis=Dabs(Enout_in(i))  !文献说chis is the ionization potential，而非能级，因此chis可能是指电离出一个电子的能量，即取Enout的绝对值
	del_s=pn_ns(nout)
	
	ksis1=2.d0*nouts1**2-pn_nss1
	
	Cs=2.4d-6*del_s*tep**(1.d0/4.d0)/chis**(7.d0/4.d0)
     &     *Dexp(-chis/tep)
	
        betas1=3.9d-28*ksis1/(chis**(7.d0/4.d0)*tep**(5.d0/4.d0))
        
        !alphas1=0给出LTE结果
        alphas1=1.9d-14*chis/tep**(1.d0/2.d0)
        alphas1=LTE_fac*alphas1 !LTE_fac=1，alphas1不变；LTE_fac=0，alphas1=0
        
        fac(i)=Cs/(ne_in*betas1+alphas1*(1.d0+dfree))
        
        term=term*fac(i)
	term_sum=term_sum+term	
		
	end do
	
	ssbrnsp(0)=1.d0/(1.d0+term_sum)
	
	ssbrnspsum=0.d0
	
	do i=0, Zan-1
	ssbrnsp(i+1)=ssbrnsp(i)*fac(i)
	ssbrnspsum=ssbrnspsum+ssbrnsp(i+1)
	end do
	ssbrnspsum=ssbrnspsum+ssbrnsp(0)
	
	sbar_out=0.d0
	
	do i=0, Zan
	sbar_out=sbar_out+ssbrnsp(i)*i
	end do
	ne_out=sbar_out*Ni
	
	del_En_out=36.d0*sbar_out*(rod/Aw)**(1.d0/3.d0)  !单位 eV
	Enout_out=Enout_pre-del_En_out
        
	end
	
	
	subroutine SNOP_nsp_jac2(Zan, n, tep, rod, Aw, Ni, dfree, 
     &                          LTE_fac, ne_in, Enout_pre, jac)
	implicit none
	
	!************  通用变量  *********************
	integer, intent(in) :: Zan, n  !元素的原子序数，束缚能层数
	real(8) :: tep, rod, Aw, Ni  !等离子体温度, 密度, 相对原子质量, 数密度
	real(8) :: dfree  !(11)式中的自由参数 d
	real(8) :: LTE_fac !控制alphas1是否为0，LTE_fac=1，alphas1不变；LTE_fac=0，alphas1=0
	real(8) :: hsp !使用二阶差分法(3点公式)计算导数时，指定的间隔
	real(8) :: Enout_pre(0:Zan)  !Enout的前半部分，跟Qnout有关，是通用的
	!********************************************
	
	!*********  此子程序所需变量  *****************
	integer :: i, j
	real(8) :: ne_in !电子数密度初值
	real(8) :: ne_in_plus, ne_in_minus
	real(8) :: ne_out_plus, ne_out_minus
	real(8) :: Enout_out_plus(0:Zan), Enout_out_minus(0:Zan)
	real(8) :: ssbrnsp_plus(0:Zan), ssbrnsp_minus(0:Zan)
	real(8) :: fne_plus, fne_minus
	real(8) :: jac
	!********************************************
	
	hsp=ne_in*1.d-5  !ne_in大概是10^(21)量级，按照以往的经验，hsp是主值的10^(-5)比较合适
	
	ne_in_plus=ne_in
	ne_in_minus=ne_in
		
	ne_in_plus=ne_in_plus+hsp  
	ne_in_minus=ne_in_minus-hsp 
	
	call SNOP_nsp2(Zan, n, tep, rod, Aw, Ni, dfree, LTE_fac,
     &       ne_in_plus, Enout_pre, ne_out_plus, Enout_out_plus,
     &       ssbrnsp_plus)  
     
	call SNOP_nsp2(Zan, n, tep, rod, Aw, Ni, dfree, LTE_fac,
     &       ne_in_minus, Enout_pre, ne_out_minus, Enout_out_minus,
     &       ssbrnsp_minus)  
     
	fne_plus=ne_in_plus-ne_out_plus
	fne_minus=ne_in_minus-ne_out_minus
	
	jac=(fne_plus-fne_minus)/(2.d0*hsp)
		
	end
	
	
	!More提出的计算平均电离度公式
	subroutine sbar_pro_More(Zan, tep, rod, Aw, sbar)
	implicit none
	
	integer :: Zan !所求元素的原子数
	real(8) :: tep, rod, Aw, sbar !等离子体温度, 密度, 相对原子质量，平均电离度
	real(8) :: R, T0, T1, alpha, beta !More公式参数
	real(8) :: a1, a2, a3, a4, AAbar !More公式参数
	real(8) :: b0, b1, b2, BB !More公式参数
	real(8) :: c1, c2, CC !More公式参数
	real(8) :: Q1, Q !More公式参数
	real(8) :: x, fx !More公式参数
	
	R=rod/(Zan*Aw)
	T0=tep/Zan**(4.d0/3.d0)
	T1=T0/(1.d0+T0)
	alpha=14.3139d0
	beta=0.6624d0
	
	a1=3.323d-3
	a2=0.971832d0
	a3=9.26184d-5
	a4=3.10165d0
	
	AAbar=a1*T0**a2+a3*T0**a4
	
	b0=-1.7630d0
	b1=1.43175d0
	b2=0.315463d0
	
	BB=-Dexp(b0+b1*T1+b2*T1**7)
	
	c1=-0.366667d0
	c2=0.983333d0
	
	CC=c1*T1+c2
	
	Q1=AAbar*R**BB
	
	Q=(R**CC+Q1**CC)**(1.d0/CC)
		
	x=alpha*Q**beta
	fx=x/(1.d0+x+Dsqrt(1.d0+2.d0*x))
	
	sbar=fx*Zan
	
	end
	
	
	!计算能级中只跟电离度有关的那部分((9)式中En的前半部分)，它们是通用的
	subroutine Enoutpre_pro2(Zan, n, tep, rod, Aw, sig, 
     &             Qnout, Enout_pre)
	implicit none
	
	integer, parameter :: max_principal = 7  ! 最大主量子数
	integer, intent(in) :: Zan, n !所求元素的原子数，束缚能层数
	real(8) :: tep, rod, Aw !等离子体温度, 密度, 相对原子质量
	real(8) :: sig(n,n)  !屏蔽系数
	integer :: i, j !计数变量i和j
	integer :: chrem, nout, pn_ns(max_principal)  !束缚电子总数，束缚电子占据的最外层主量子数，s度电离情况下各层束缚电子数
	integer :: nouts1, pn_nss1 !s+1度电离时束缚电子占据的最外层主量子数，s+1度电离时有束缚电子的最外层的电子数
	real(8) :: pmsig  !(8)式等号右边Z减去的两项
	real(8) :: Qnout(0:Zan), Enout_pre(0:Zan) !不同电离度情况下最外层的有效电荷, 能级中只跟电离度有关的部分
	
	Qnout=0.d0
	Enout_pre=0.d0
	nouts1=0
	pn_nss1=0
	
	do i=0, Zan !电离度从0到Zan
	chrem=Zan-i !剩余电荷
	
	call pn_ns_Z2(Zan, n, chrem, pn_ns, nout, nouts1, 
     &                pn_nss1)
	
	pmsig=0.d0
		
	!计算最外层nout电子的Qn
	do j=1, nout
	   if(j .eq. nout) then
	   pmsig=pmsig
	   else
	   pmsig=pmsig+pn_ns(j)*sig(nout,j)
	   end if
	end do 
	pmsig=pmsig+pn_ns(nout)*sig(nout,nout)*1.d0/2.d0
	Qnout(i)=Zan-pmsig  !公式(8)
	
	Enout_pre(i)=13.6d0*Qnout(i)**2/nout**2 !Enout的前半部分，只跟电离度有关，是通用的。后面算出sbar_nsp之后可以算新的del_En_nsp，从而给出新的Enout_nsp作为输出值
	end do
	
	end
	
	subroutine abs_coe4(Zan, n, tep, rod, Aw, Ni, sig, fnm, 
     &             nsp_in, Qnout_in, Enout_in, ne_in, hplnu, lbfac,
     &             fffac, bffac, bbfac, esfac, emitot, abstot)
	implicit none
	
	integer, parameter :: max_principal = 7  ! 最大主量子数
	integer, intent(in) :: Zan !原子序数
	integer, intent(in) :: n !原子核外能层数
	real(8) :: hplnu, ne_in, Ni, tep, rod, Aw !光子能量，自由电子数密度，等离子体数密度，温度，密度，相对原子质量
	real(8) :: sig(n,n), fnm(n-1,n) !屏蔽系数，hydrogen oscillator strength fnm
	real(8) :: nsp_in(0:Zan), Enout_in(0:Zan) !不同电离度离子的数密度概率，不同电离度的能量
	real(8) :: x, ns_in(0:Zan) !hplnu/kb*T，不同电离度离子的数密度
	real(8) :: emiff_pre1, emiff_pre2, emiff !ff过程的发射吸收
	real(8) :: InuP, absff !Planck function，ff过程吸收系数
	integer :: i, j, k
	real(8) :: emibf_pre1, emibf_pre2, emibf !bf过程的发射系数
	integer :: chrem, pn_ns(max_principal), nout !s度电离时束缚电子数，每层束缚电子数，有电子占据的最外层
	integer :: nouts1, pn_nss1 !s+1度电离时s度电离的参数
	real(8) :: chis, ksis1 !Enout_in绝对值，
	real(8) :: Hstp !阶跃函数
	real(8) :: Qnout_in(0:Zan) !不同电离度情况下最外层的有效电荷
	real(8) :: absbf_pre1, absbf_pre2, absbf_pre3, absbf !bf过程的吸收系数
	real(8) :: Qnnbf(n), Ennbf(n) !不同电离度情况下n层的有效电荷, n能层的能级
	real(8) :: lsum, lsum_pre !sum over all lines considered
	real(8) :: Qnnbb(n), Ennbb(n) !不同电离度情况下n层的有效电荷, n能层的能级
	real(8) :: wlb, lbfac, FFnm, del_chi, Llp !公式(28)，wlb放大系数，F_nm，E(n,s)-E(m,s)，公式(27)
	real(8), parameter :: pi=4.d0*DATAN(1.d0) !圆周率 pi
	real(8) :: emibb_B, emibb_C, emibb !bb过程的发射系数
	real(8) :: absbb !bb过程的吸收系数
	real(8) :: emitot, abstot !3种过程总的发射系数，总的吸收系数
	real(8) :: fffac, bffac, bbfac !控制emitot和abstot里面显示哪些项，例如fffac=1表示有ff过程的贡献，0则表示没有，这样可以单独得到某个过程的贡献
	integer :: s_ini !不同元素在bf过程中的起始电离度
	real(8) :: esfac, abses !es过程是否计算控制系数，es过程吸收系数
	
	x=hplnu/tep
	ns_in=nsp_in*Ni !注意后面的公式中，有的需要nsp，有的需要ns
	
	!f-f过程的发射和吸收系数计算
	emiff_pre1=5.d-41*ne_in*tep**(-1.d0/2.d0)*dexp(-x)
	
	emiff_pre2=0.d0
	do i=0, Zan
	emiff_pre2=emiff_pre2+ns_in(i)*i**2 !注意是ns_in
	end do
	emiff=emiff_pre1*emiff_pre2
	
	!print*, 'emiff', emiff
	
	InuP=2.07d-4*hplnu**3/(dexp(x)-1.d0)
	absff=emiff/(rod*InuP)
	
	!b-f过程的发射和吸收系数计算
	emibf_pre1=8.d-40*ne_in/tep**(3.d0/2.d0)*dexp(-x)

	!print*, 'emibf_pre1', emibf_pre1
	
	emibf_pre2=0.d0
	
	!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	!此处跟具体元素的核外电子排布有关
	!对于bf过程，不同元素有不同的起始电离度:
	!Be为0；C为0；Al为0；Ti为2；Fe为2；Cu为0；Mo为1；Sn为4；Ba为10；Eu为10；W为2；Au为1；Pb为4；U为11。
	!这里给出起始电离度s_ini
	select case(Zan)
	case(4) !Be
	  s_ini=0
	case(6) !C
	  s_ini=0
	case(13) !Al
	  s_ini=0
	case(22) !Ti
	  s_ini=2
	case(26) !Fe
	  s_ini=2
	case(29) !Cu
	  s_ini=0
	case(42) !Mo
	  s_ini=1
	case(50) !Sn
	  s_ini=4
	case(56) !Ba
	  s_ini=10
	case(63) !Eu
	  s_ini=10
	case(74) !W
	  s_ini=2
	case(79) !Au
	  s_ini=1
	case(82) !Pb
	  s_ini=4
	case(92) !U
	  s_ini=11
	case default
	  print*, 'Error: No code for element with atomic number', Zan
	  stop
	end select
	!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	
	do i=s_ini, Zan-1
	chrem=Zan-i !剩余电荷
	call pn_ns_Z2(Zan, n, chrem, pn_ns, nout, nouts1, 
     &               pn_nss1)
	
	chis=Dabs(Enout_in(i))
	
	ksis1=2.d0*nouts1**2-pn_nss1
	
	if(hplnu-chis .lt. 0.d0) then
	Hstp=0.d0
	else
	Hstp=1.d0
	end if
	
	!注意此处是ns_in
	emibf_pre2=emibf_pre2+ns_in(i+1)*ksis1/nout**5*Qnout_in(i)**4
     &             *dexp(chis/tep)*Hstp 
     
	end do
	
	emibf=emibf_pre1*emibf_pre2
	
	
	absbf_pre1=1.2d10/Aw*(1.d0-dexp(-x))
	
	absbf_pre2=0.d0
	
	do i=s_ini, Zan !i对电离度 s 循环，j对能层 n 循环
	chrem=Zan-i !剩余电荷
	
	call pn_ns_Z2(Zan, n, chrem, pn_ns, nout, nouts1, 
     &               pn_nss1)
	call Enn_pro2(Zan, n, rod, Aw, Ni, sig, ne_in, chrem, 
     &               Qnnbf, Ennbf)
	
	absbf_pre3=0.d0
	do j=1, n	
	if(hplnu-Ennbf(j) .lt. 0.d0) then
	Hstp=0.d0
	else
	Hstp=1.d0
	end if
	
	absbf_pre3=absbf_pre3+Qnnbf(j)**4*pn_ns(j)/(j**5*hplnu**3)
     &             *Hstp
	end do
	absbf_pre2=absbf_pre2+nsp_in(i)*absbf_pre3 !注意是nsp_in
		
	end do
	
	absbf=absbf_pre1*absbf_pre2
	
	!b-b过程的发射和吸收系数计算
	lsum=0.d0
	
	do i=s_ini, Zan !i对电离度s循环，j对n循环，k对m循环
	  chrem=Zan-i !剩余电荷
	  
	  call pn_ns_Z2(Zan, n, chrem, pn_ns, nout, nouts1, 
     &                 pn_nss1)
	  
	  call Enn_pro2(Zan, n, rod, Aw, Ni, sig, ne_in, chrem, 
     &                 Qnnbb, Ennbb)
	  lsum_pre=0.d0
	  do j=nout-1, nout
	    if(j .lt. 1) then 
	      lsum_pre=lsum_pre
	      cycle
	    else
	      if(j==n) then  !此种j的限制意味着Al电离度为0、1、2时，只有nout-1的能级有贡献
	        lsum_pre=lsum_pre
	        exit !目前的想法是，n*=nmax可以算n=nmax-1，m=nmax的贡献，也就是说无非这种电离度的贡献只有一部分罢了。
	           !如果后续改为不考虑此种电离度s，那么判断语句需要改为 if(nout==nmax) 
	      else
	        wlb=0.637d-22*ne_in/dsqrt(tep)*(5.d0*j**2/Qnnbb(j)**2*
     &              ((i+1.d0)**2*j**2/Qnnbb(j)**2-1.d0))
	        wlb=lbfac*wlb !lbfac是人为引入的线数量放大系数，文献给了lbfac=1000的图像
	      
	        do k=j+1, n
	          FFnm=fnm(j, k)*pn_ns(j)*(1.d0-pn_ns(k)/(2.d0*k**2))
	          del_chi=Ennbb(j)-Ennbb(k) 
	          Llp=1.d0/pi*wlb/((del_chi-hplnu)**2+wlb**2)
	          lsum_pre=lsum_pre+FFnm*Llp
	        end do
	      end if
	    end if
	  end do
	  lsum=lsum+nsp_in(i)*lsum_pre  !注意是nsp(s)
	end do
	
	emibb_B=2.3d-20*Ni*dexp(-x)*hplnu**3*lsum
	emibb_C=2.3d-33*ne_in*Ni*tep**(1.d0/2.d0)*lsum
	emibb=1.d0/(1.d0/emibb_B+1.d0/emibb_C)
	
	absbb=6.6d7*(1.d0-dexp(-x))/Aw*lsum
	
	!es过程的吸收系数计算
	abses=0.4d0*ne_in/Ni/Aw
	
	!4种过程总的发射和吸收系数
	emitot=fffac*emiff+bffac*emibf+bbfac*emibb
	abstot=fffac*absff+bffac*absbf+bbfac*absbb+esfac*abses
	
	end
	
	
	!bf、bb过程需要s电离度下每个能层的Qn和En，这里计算
	!对于bf过程，(21)式对n的求和按说应到nmax，但是由于n大于nout的Pn(s)=0，所以上限可以取nout
	!对于bb过程，则是由于m可以取nmax，因此n的上限应取nmax
	!综上，n的上限取nmax，对于bf和bb都是正确的
	subroutine Enn_pro2(Zan, n, rod, Aw, Ni, sig, ne_in, 
     &                     chrem, Qnn, Enn)
	implicit none
	
	integer, parameter :: max_principal = 7  ! 最大主量子数
	integer, intent(in) :: Zan, n !所求元素的原子数, 束缚能层数
	real(8) :: rod, Aw, Ni !等离子体密度, 相对原子质量，数密度
	real(8) :: sig(n,n)  !屏蔽系数
	integer :: i, j !计数变量i和j
	integer :: chrem, nout, pn_ns(max_principal)  !束缚电子总数，束缚电子占据的最外层主量子数，s度电离情况下各层束缚电子数
	integer :: nouts1, pn_nss1 !s+1度电离时束缚电子占据的最外层主量子数，s+1度电离时有束缚电子的最外层的电子数
	real(8) :: pmsig  !(8)式等号右边Z减去的两项
	real(8) :: Qnn(n), Enn_pre(n), Enn(n) !不同电离度情况下n层的有效电荷, 能级中只跟电离度有关的部分，n能层的能级
	real(8) :: ne_in, sbar_in, del_En
	
	sbar_in=ne_in/Ni
	del_En=36.d0*sbar_in*(rod/Aw)**(1.d0/3.d0)  !单位 eV
	
	call pn_ns_Z2(Zan, n, chrem, pn_ns, nout, nouts1, 
     &               pn_nss1)
	
	Qnn=0.d0
	Enn_pre=0.d0
	Enn=0.d0	
	!计算各能层的Qn
	do i=1, n
	pmsig=0.d0
	do j=1, i
	   if(j .eq. i) then
	   pmsig=pmsig
	   else
	   pmsig=pmsig+pn_ns(j)*sig(i,j)
	   end if
	end do 
	
	pmsig=pmsig+pn_ns(i)*sig(i,i)*1.d0/2.d0
	Qnn(i)=Zan-pmsig  !公式(8)
	
	Enn_pre(i)=13.6d0*Qnn(i)**2/i**2 !
	Enn(i)=abs(Enn_pre(i)-del_En) !E(n,s)应该同chis一样，都是绝对值
	end do
	
	end
	
	
	subroutine pn_ns_Z2(Zan, n, chrem, pn, nout, nouts1, pn_nss1)
	implicit none
	
	integer :: Zan, n !原子序数，主量子数
	integer, parameter :: max_orbitals = 19  ! 子壳层总数
	integer, parameter :: max_principal = 7  ! 最大主量子数
	integer :: orb_n(max_orbitals), orb_l(max_orbitals)  
	integer :: orb_cap(max_orbitals)  
	integer :: elec_neutral(max_orbitals)  ! 中性原子的子壳层电子数
	integer :: elec_ion(max_orbitals)  ! 离子的子壳层电子数
	integer :: layer_count(max_principal)  ! 每层包含的子壳层列表（按 l 降序排列）
	integer :: layer_list(max_principal, 10)  ! 每层最多10个子壳层
	integer :: pn(max_principal)  ! 每层总电子数
	integer :: s, nout, chrem !电离度，最外层主量子数，剩余电荷
	integer :: i, j, rem
	integer :: nouts1, pn_nss1 !s+1度电离时束缚电子占据的最外层主量子数，s+1度电离时有束缚电子的最外层的电子数
	
	!------------------ 初始化子壳层数据 ------------------
	! 顺序严格按照能量从低到高（Madelung 规则）
	! 索引: 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
	!原子电子层排布遵循构造原理（Aufbau principle），按照能级顺序填充：
	!1s,2s,2p,3s,3p,4s,3d,4p,5s,4d,5p,6s,4f,5d,6p,7s,5f,6d,7p... 
	!每层（主量子数n）包含s,p,d,f等亚层，每个亚层最多电子数：s=2, p=6, d=10, f=14
	data orb_n /1, 2, 2, 3, 3, 4, 3, 4, 5, 4, 5, 6, 4, 5, 6, 
     &              7, 5, 6, 7/  !1s,2s,2p,3s,3p,4s,3d,4p,5s,4d,5p,6s,4f,5d,6p,7s,5f,6d,7p
	data orb_l /0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 
     &              0, 3, 2, 1/  !s, l=0; p, l=1, d, l=2; f, l=3，角量子数
	data orb_cap /2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 
     &                10, 6, 2, 14, 10, 6/ !orb_n里面的各能级所包含的电子数
     
     
	! 构建每层子壳层列表（l 降序，用于电离时优先失去高 l 电子）
	layer_count(1) = 1
	layer_list(1,1) = 1  !1s
	
	layer_count(2) = 2
	layer_list(2,1) = 3  !2p
	layer_list(2,2) = 2  !2s
	
	layer_count(3) = 3 
	layer_list(3,1) = 7  !3d
	layer_list(3,2) = 5  !3p
        layer_list(3,3) = 4  !3s
     
	layer_count(4) = 4  
	layer_list(4,1) =13  !4f
	layer_list(4,2) =10  !4d
        layer_list(4,3) =8  !4p
	layer_list(4,4) =6  !4s
	
	layer_count(5) = 4  
	layer_list(5,1) =17  !5f
	layer_list(5,2) =14  !5d
        layer_list(5,3) =11  !5p
	layer_list(5,4) =9  !5s
	
	layer_count(6) = 3  
	layer_list(6,1) =18  !6d
	layer_list(6,2) =15  !6p
        layer_list(6,3) =12  !6s
        
	layer_count(7) = 2 
	layer_list(7,1) =19  !7p
	layer_list(7,2) =16  !7s
	
	if (Zan <= 0 .or. Zan > 118) then
	  print *, '错误：Z 必须在 1~118 之间。'
	  stop
	end if
	if (n < 1 .or. n > max_principal) then
	  print *, '错误：n 必须在 1~7 之间。'
          stop
	end if
	
	!------------------ 计算中性原子电子排布 ------------------
	elec_neutral = 0
	rem = Zan
	do i = 1, max_orbitals
          if (rem <= 0) exit
          elec_neutral(i) = min(rem, orb_cap(i))
          rem = rem - elec_neutral(i)
	end do
	
	! 修正已知的基态例外情况
	select case (Zan)
	case (24)      ! Cr: 3d4 4s2 -> 3d5 4s1
	  elec_neutral(6) = elec_neutral(6) - 1
	  elec_neutral(7) = elec_neutral(7) + 1
	case (29)      ! Cu: 3d9 4s2 -> 3d10 4s1
          elec_neutral(6) = elec_neutral(6) - 1
          elec_neutral(7) = elec_neutral(7) + 1
	case (41)      ! Nb: 4d3 5s2 -> 4d4 5s1
	  elec_neutral(9) = elec_neutral(9) - 1
	  elec_neutral(10) = elec_neutral(10) + 1
	case (42)      ! Mo: 4d4 5s2 -> 4d5 5s1
          elec_neutral(9) = elec_neutral(9) - 1
	  elec_neutral(10) = elec_neutral(10) + 1
	case (44)      ! Ru: 4d6 5s2 -> 4d7 5s1
	  elec_neutral(9) = elec_neutral(9) - 1
	  elec_neutral(10) = elec_neutral(10) + 1
	case (45)      ! Rh: 4d7 5s2 -> 4d8 5s1
	  elec_neutral(9) = elec_neutral(9) - 1
	  elec_neutral(10) = elec_neutral(10) + 1
	case (46)      ! Pd: 4d8 5s2 -> 4d10
	  elec_neutral(9) = elec_neutral(9) - 2
	  elec_neutral(10) = elec_neutral(10) + 2
	case (47)      ! Ag: 4d9 5s2 -> 4d10 5s1
	  elec_neutral(9) = elec_neutral(9) - 1
	  elec_neutral(10) = elec_neutral(10) + 1
	case (57)      ! La: 4f1 6s2 -> 5d1 6s2
	  elec_neutral(13) = elec_neutral(13) - 1
	  elec_neutral(14) = elec_neutral(14) + 1
	case (58)      ! Ce: 4f2 6s2 -> 4f1 5d1 6s2
	  elec_neutral(13) = elec_neutral(13) - 1
	  elec_neutral(14) = elec_neutral(14) + 1
	case (64)      ! Gd: 4f8 6s2 -> 4f7 5d1 6s2
	  elec_neutral(13) = elec_neutral(13) - 1
	  elec_neutral(14) = elec_neutral(14) + 1
	case (78)      ! Pt: 5d8 6s2 -> 5d9 6s1
	  elec_neutral(12) = elec_neutral(12) - 1
	  elec_neutral(14) = elec_neutral(14) + 1
	case (79)      ! Au: 5d9 6s2 -> 5d10 6s1
	  elec_neutral(12) = elec_neutral(12) - 1
	  elec_neutral(14) = elec_neutral(14) + 1
	case (89)      ! Ac: 5f1 7s2 -> 6d1 7s2
	  elec_neutral(17) = elec_neutral(17) - 1
	  elec_neutral(18) = elec_neutral(18) + 1
	case (90)      ! Th: 5f2 7s2 -> 6d2 7s2
	  elec_neutral(17) = elec_neutral(17) - 2
	  elec_neutral(18) = elec_neutral(18) + 2
	case (91)      ! Pa: 5f3 7s2 -> 5f2 6d1 7s2
	  elec_neutral(17) = elec_neutral(17) - 1
	  elec_neutral(18) = elec_neutral(18) + 1
	case (92)      ! U: 5f4 7s2 -> 5f3 6d1 7s2
	  elec_neutral(17) = elec_neutral(17) - 1
	  elec_neutral(18) = elec_neutral(18) + 1
	case (93)      ! Np: 5f5 7s2 -> 5f4 6d1 7s2
	  elec_neutral(17) = elec_neutral(17) - 1
	  elec_neutral(18) = elec_neutral(18) + 1
	case (96)      ! Cm: 5f8 7s2 -> 5f7 6d1 7s2
	  elec_neutral(17) = elec_neutral(17) - 1
	  elec_neutral(18) = elec_neutral(18) + 1
	end select
	
	!------------------ 输入电离度并输出排布 ------------------
	s=Zan-chrem
	
	if (s < 0) stop
	if (s > Zan) then
	  print *, '电离度不能超过原子序数，请重新输入'
          stop
        end if
          
        ! 复制中性排布并移除电子
	elec_ion = elec_neutral
	call remove_electrons(elec_ion, s, layer_count, layer_list, 
     &                        max_principal, max_orbitals)
     
	! 计算每层总电子数
	pn = 0
	nout = 0
	do i = 1, max_orbitals
          j = orb_n(i)
          pn(j) = pn(j) + elec_ion(i)
          if (elec_ion(i) > 0 .and. j > nout) nout = j
        end do
	
	if(s==Zan) nout=1
	
	nouts1=nout
	pn_nss1=max(pn(nout)-1, 0) !最小值是0
  
	end
	
	!----------------------------------------------------------------
	! 子程序：从离子电子组态中移除 s 个电子
	! 规则：从最外层（最大 n）开始，同层内先失去 l 最大的电子
	!----------------------------------------------------------------
	subroutine remove_electrons(elec, s, layer_count, layer_list, 
     &                              maxp, maxorb)
	implicit none
	
	integer, intent(inout) :: elec(maxorb)
	integer, intent(in)    :: s
	integer, intent(in)    :: maxp
	integer, intent(in)    :: layer_count(maxp)
	integer, intent(in)    :: layer_list(maxp, 10)
	integer, intent(in)    :: maxorb
	integer :: ns, i, idx, remove, remain

	remain = s
	do ns = maxp, 1, -1
          if (remain <= 0) exit
            do i = 1, layer_count(ns)
              idx = layer_list(ns, i)
                if (elec(idx) > 0) then
                  remove = min(remain, elec(idx))
                  elec(idx) = elec(idx) - remove
                  remain = remain - remove
                  if (remain == 0) return
                end if
            end do
	end do
	end 
	
	end module
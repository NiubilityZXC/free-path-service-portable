module gpu_bridge
  use, intrinsic :: iso_c_binding
  implicit none

  interface
	    integer(c_int) function ross_gpu_init(zan, n, tep, rod, aw, ni, &
	     & sig, fnm, nsp, qnout, enout, ne, tgama, lbfac, fffac, bffac, &
	     & bbfac, esfac) &
	     & bind(C, name="ross_gpu_init")
      import :: c_int, c_double
      integer(c_int), value :: zan, n
	      real(c_double), value :: tep, rod, aw, ni, ne, tgama, lbfac
	      real(c_double), value :: fffac, bffac, bbfac, esfac
      real(c_double), intent(in) :: sig(*), fnm(*), nsp(*)
      real(c_double), intent(in) :: qnout(*), enout(*)
    end function ross_gpu_init

	    integer(c_int) function ross_gpu_eval(nvec, x, f, core) &
	     & bind(C, name="ross_gpu_eval")
	      import :: c_int, c_double
	      integer(c_int), value :: nvec
	      real(c_double), intent(in) :: x(*)
	      real(c_double), intent(out) :: f(*)
	      integer(c_int), value :: core
	    end function ross_gpu_eval
  end interface

contains

	  integer function init_gpu_bridge(zan, n, tep, rod, aw, ni, sig, fnm, &
	   & nsp, qnout, enout, ne, tgama, lbfac, fffac, bffac, bbfac, &
	   & esfac) result(status)
    integer, intent(in) :: zan, n
	    real(8), intent(in) :: tep, rod, aw, ni, ne, tgama, lbfac
	    real(8), intent(in) :: fffac, bffac, bbfac, esfac
    real(8), intent(in) :: sig(*), fnm(*), nsp(*), qnout(*), enout(*)

    status = ross_gpu_init(int(zan, c_int), int(n, c_int), tep, rod, aw, &
	   &                       ni, sig, fnm, nsp, qnout, enout, ne, tgama, &
	   &                       lbfac, fffac, bffac, bbfac, esfac)
    if (status == 1) then
      print *, '[gpu] 已配置 CUDA 批量 integrand（worker 内懒初始化）'
    else
      print *, '[gpu] CUDA 配置未启用，使用 CPU 路径'
    end if
    flush(6)
  end function init_gpu_bridge

	  integer function eval_gpu_bridge(nvec, x, f, core) result(status)
	    integer, intent(in) :: nvec, core
	    real(8), intent(in) :: x(*)
	    real(8), intent(out) :: f(*)

	    status = ross_gpu_eval(int(nvec, c_int), x, f, int(core, c_int))
	  end function eval_gpu_bridge

end module gpu_bridge

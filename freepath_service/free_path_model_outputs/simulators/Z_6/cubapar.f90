module cubapar
  implicit none

contains

  subroutine count_cpu_token(tok, nadd)
    character(*), intent(in) :: tok
    integer, intent(out) :: nadd
    integer :: dash, lo, hi, ios

    nadd = 0
    dash = index(trim(tok), '-')
    if (dash > 0) then
      read(tok(:dash-1), *, iostat=ios) lo
      if (ios /= 0) return
      read(tok(dash+1:), *, iostat=ios) hi
      if (ios /= 0) return
      if (hi >= lo) nadd = hi - lo + 1
    else
      read(tok, *, iostat=ios) lo
      if (ios == 0) nadd = 1
    end if
  end subroutine count_cpu_token

  integer function count_cpu_list(list) result(n)
    character(*), intent(in) :: list
    character(len=512) :: part
    integer :: pos, last, nadd

    n = 0

    last = 1
    do
      pos = index(list(last:), ',')
      if (pos == 0) then
        part = adjustl(list(last:))
        call count_cpu_token(part, nadd)
        n = n + nadd
        exit
      else
        part = adjustl(list(last:last + pos - 2))
        call count_cpu_token(part, nadd)
        n = n + nadd
        last = last + pos
      end if
    end do
  end function count_cpu_list

  integer function get_ncpus_online() result(n)
    character(len=512) :: list
    integer :: u, ios

    n = 0
    open(newunit=u, file='/sys/devices/system/cpu/online', &
     &   status='old', action='read', iostat=ios)
    if (ios /= 0) return

    read(u, '(A)', iostat=ios) list
    close(u)
    if (ios /= 0) return

    n = count_cpu_list(adjustl(list))
  end function get_ncpus_online

  integer function get_ncpus_affinity() result(n)
    character(len=512) :: line, list
    integer :: u, ios

    n = 0
    open(newunit=u, file='/proc/self/status', status='old', &
     &   action='read', iostat=ios)
    if (ios /= 0) return

    do
      read(u, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (index(line, 'Cpus_allowed_list:') /= 1) cycle

      list = adjustl(line(19:))
      n = count_cpu_list(list)
      exit
    end do

    close(u)
  end function get_ncpus_affinity

  subroutine setup_cuba_parallel(use_gpu)
    integer, intent(in) :: use_gpu
    character(len=64) :: val
    integer :: stat, lenval, ios, ncores, pcores
    integer :: naccel, paccel
    external cubacores, cubaaccel

    ncores = 0

    call get_environment_variable('CUBACORES', val, length=lenval, &
     &                            status=stat)
    if (stat == 0 .and. lenval > 0) then
      print *, '[cubapar] CUBACORES=', trim(val), &
     &        ' (使用环境变量设置的 Cuba 核数)'
      flush(6)
      ncores = -1
    end if

    if (ncores /= -1) then
      call get_environment_variable('ROSS_CUBACORES', val, length=lenval, &
       &                            status=stat)
      if (stat == 0 .and. lenval > 0) then
        read(val(:lenval), *, iostat=ios) ncores
        if (ios /= 0) ncores = 0
      else
        ncores = get_ncpus_online()
        if (ncores < 1) ncores = get_ncpus_affinity()
      end if

      if (ncores < 1) ncores = 1
      pcores = 10000

      call cubacores(ncores, pcores)
      print *, '[cubapar] 已设置 Cuba CPU 核数 =', ncores
    end if

    naccel = 0
    paccel = 0
    if (use_gpu == 1) then
      naccel = 1
      call get_environment_variable('ROSS_CUBAACCELS', val, &
       &                            length=lenval, status=stat)
      if (stat == 0 .and. lenval > 0) then
        read(val(:lenval), *, iostat=ios) naccel
        if (ios /= 0 .or. naccel < 1) naccel = 1
      end if
      if (naccel > 1) then
        paccel = 1000000
      else
        paccel = 1750000
      end if

      call get_environment_variable('ROSS_CUBAACCELMAX', val, &
       &                            length=lenval, status=stat)
      if (stat == 0 .and. lenval > 0) then
        read(val(:lenval), *, iostat=ios) paccel
        if (ios /= 0 .or. paccel < 1) then
          if (naccel > 1) then
            paccel = 1000000
          else
            paccel = 1750000
          end if
        end if
      end if
    end if

    call cubaaccel(naccel, paccel)
    if (naccel > 0) then
      print *, '[cubapar] 已启用 Cuba GPU accelerator 数 =', naccel, &
     &         ' 每批点数 =', paccel
    else
      print *, '[cubapar] 未启用 Cuba GPU accelerator'
    end if
    flush(6)
  end subroutine setup_cuba_parallel

end module cubapar

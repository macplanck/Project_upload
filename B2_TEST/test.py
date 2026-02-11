func_chosen = 2

if func_chosen == 1:
    from B2_TEST.test_lib_1 import func
else:
    from B2_TEST.test_lib_2 import func


if __name__ == '__main__':
    func()
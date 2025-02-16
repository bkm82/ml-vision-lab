from ml_vision_lab.foo import foo, my_add


def test_foo():
    assert foo("foo") == "foo"


def test_my_add():
    a = 5
    b = 7
    actual = my_add(a=a, b=b)
    expected = 12
    assert actual == expected

import torch

from rewind import hashing


def test_deterministic():
    t = torch.arange(1000, dtype=torch.float32)
    assert hashing.fingerprint_int(t) == hashing.fingerprint_int(t.clone())


def test_bitflip_changes_digest():
    t = torch.arange(1000, dtype=torch.float32)
    u = t.clone()
    u.view(torch.int32)[500] ^= 1  # exactly one bit
    assert hashing.fingerprint_int(t) != hashing.fingerprint_int(u)


def test_order_sensitive():
    t = torch.arange(1000, dtype=torch.float32)
    perm = t.flip(0)
    assert hashing.fingerprint_int(t) != hashing.fingerprint_int(perm)


def test_paired_swap_changes_digest():
    # plain XOR folding cancels swapped identical words; position salt must not
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    u = torch.tensor([2.0, 1.0, 3.0, 4.0])
    assert hashing.fingerprint_int(t) != hashing.fingerprint_int(u)


def test_dtype_salted():
    a = torch.zeros(16, dtype=torch.float32)
    b = torch.zeros(32, dtype=torch.float16)  # same byte count, same bytes
    assert hashing.fingerprint_int(a) != hashing.fingerprint_int(b)


def test_shape_salted():
    a = torch.zeros(4, 8)
    b = torch.zeros(8, 4)
    assert hashing.fingerprint_int(a) != hashing.fingerprint_int(b)


def test_empty_and_scalar():
    assert isinstance(hashing.fingerprint_int(torch.empty(0)), int)
    assert isinstance(hashing.fingerprint_int(torch.tensor(3.14)), int)


def test_non_contiguous():
    t = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    assert hashing.fingerprint_int(t[:, 3]) == hashing.fingerprint_int(
        t[:, 3].contiguous()
    )


def test_misaligned_offset_view():
    # x[1:] is contiguous but starts 4 bytes into storage
    t = torch.arange(9, dtype=torch.float32)
    assert hashing.fingerprint_int(t[1:]) == hashing.fingerprint_int(t[1:].clone())


def test_bool_does_not_collide_with_uint8():
    a = torch.tensor([True, False, True])
    b = torch.tensor([1, 0, 1], dtype=torch.uint8)
    assert hashing.fingerprint_int(a) != hashing.fingerprint_int(b)


def test_odd_byte_lengths():
    for n in (1, 3, 7, 9):
        t = torch.arange(n, dtype=torch.uint8)
        assert isinstance(hashing.fingerprint_int(t), int)


def test_bool_tensor():
    t = torch.tensor([True, False, True])
    assert hashing.fingerprint_int(t) == hashing.fingerprint_int(t.clone())


def test_hex_digest():
    assert hashing.hex_digest(-1) == "f" * 16
    assert len(hashing.hex_digest(12345)) == 16


def test_combine_ints_order_sensitive():
    assert hashing.combine_ints([1, 2]) != hashing.combine_ints([2, 1])
    assert hashing.combine_ints([]) == hashing.combine_ints([])

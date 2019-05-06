import TensorFlow

var x: Tensor<Float> = [[
    [1, 2, 4],
    [4, 5, 6],
]]

func at(_ n: Int) -> TensorRange {
    return .index(n)
}

func all(_ stride: Int = 1) -> TensorRange {
    return .partialRangeFrom(0..., stride: stride)
}

func from(_ n: Int, _ stride: Int = 1) -> TensorRange {
    return .partialRangeFrom(n..., stride: stride)
}

func upTo(_ n: Int, _ stride: Int = 1) -> TensorRange {
    return .partialRangeUpTo(..<n, stride: stride)
}

func range(_ from: Int, _ to: Int, _ stride: Int = 1) -> TensorRange {
    return .range(from ..< to, stride: stride)
}

// x = x[TensorRange.partialRangeFrom(0..., stride: 1), TensorRange.range(0 ..< 1, stride: 1)]
x = x[at(-1), all(), range(-2, -1)]

print(x)

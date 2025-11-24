#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"
#include "cutlass/epilogue/thread/activation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass {
namespace epilogue {
namespace thread {

template <typename A, typename B = A>
struct silu_multiply {
  CUTLASS_HOST_DEVICE
  B operator()(A const &a, B const &b) const {
    cutlass::epilogue::thread::SiLu<B> silu;
    return silu(B(a)) * b;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = Silu(accumulator) * beta * source
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation.
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename ElementSource_ = ElementOutput_
>
class CustomSwiGluCombination {
public:

  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementC = ElementSource_;
  using ElementD = ElementOutput_;

  static int const kCount = Count;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params 
  {
    ElementCompute beta;

    CUTLASS_HOST_DEVICE
      Params() : beta(ElementCompute(0)) { }

    explicit CUTLASS_HOST_DEVICE
    Params(
      ElementCompute beta
    ):
      beta(beta) { }
  };

private:

  //
  // Data members
  //

  ElementCompute beta_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  explicit CustomSwiGluCombination(Params const &params, int group_idx) {
    // printf("CustomSwiGluCombination::params.beta[%f]\r\n", params.beta);
    beta_ = params.beta;
  }

  CUTLASS_HOST_DEVICE
  explicit CustomSwiGluCombination(const Params & params) 
  : CustomSwiGluCombination(params, /* group_idx */ 0) { }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    // printf("is_source_needed::beta_[%f]\r\n", beta_);
    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  /// Computes linear scaling with source: D = SiLU(accumulator) * beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator,
      FragmentSource const &source) const {

    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    silu_multiply<FragmentCompute> silu_multiply_accumulator;

    intermediate = mul_add_source(beta_, converted_source);                             // intermediate = beta_ * converted_source

    intermediate = silu_multiply_accumulator(converted_accumulator, intermediate);    // D = SiLu(accum) * intermediate

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    return destination_converter(converted_accumulator);
  }

  //
  // Specializations for scalar (for use with cute::collective::DefaultEpilogue)
  //
  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator, ElementC const source) const {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    [[maybe_unused]] NumericConverter<ElementCompute, ElementC, Round> source_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;

    // Convert to destination numeric type

    ElementCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    ElementCompute intermediate;
    multiplies<ElementCompute> multiply;
    silu_multiply<ElementCompute> silu_mul;

    intermediate = multiply(beta_, source);                            // X =  beta_ * source

    intermediate = silu_mul(converted_accumulator, intermediate);    // D =  SiLu(Accum) * X
    return destination_converter(intermediate);
  }

  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator) const {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;
    ElementCompute converted_accumulator = accumulator_converter(accumulator);

    return destination_converter(converted_accumulator);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

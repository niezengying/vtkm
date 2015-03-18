//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 Sandia Corporation.
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2014. Los Alamos National Security
//
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================
#ifndef vtk_m_exec_cuda_internal_ArrayPortalFromThrustTransposed_h
#define vtk_m_exec_cuda_internal_ArrayPortalFromThrustTransposed_h

#include <vtkm/Types.h>

#include <iterator>

// Disable GCC warnings we check vtkmfor but Thrust does not.
#if defined(__GNUC__) && !defined(VTKM_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#endif // gcc version >= 4.6
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 2)
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif // gcc version >= 4.2
#endif // gcc && !CUDA

#include <thrust/system/cuda/memory.h>

#if defined(__GNUC__) && !defined(VTKM_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif // gcc version >= 4.6
#endif // gcc && !CUDA

//included Since we are a portal of portals!
#include <vtkm/exec/cuda/internal/ArrayPortalFromThrust.h>

namespace vtkm {
namespace exec {
namespace cuda {
namespace internal {

template<typename VecType>
struct to_value_type
{
  typedef typename VecType::ComponentType ComponentType;
  static const vtkm::IdComponent NUM_COMPONENTS = VecType::NUM_COMPONENTS;

  template< typename IteratorVecType >
  VTKM_EXEC_EXPORT
  static VecType get(vtkm::Id index, const IteratorVecType& iterators)
  {
  VecType v;
  #pragma unroll
  for(int i=0; i < NUM_COMPONENTS; ++i)
    { v[i] = load_through_texture< const ComponentType >::get( iterators[i]+index ); }
  return v;
  }

  template< typename IteratorVecType >
  VTKM_EXEC_EXPORT
  static void set(vtkm::Id index, const VecType& value,  IteratorVecType& iterators)
  {
  #pragma unroll
  for(int i=0; i < NUM_COMPONENTS; ++i)
    { *(iterators[i]+index) = value[i]; }
  }
};


/// This templated implementation of an ArrayPortal allows you to adapt a pair
/// of begin/end iterators to an ArrayPortal interface.
///
template<typename T>
class ArrayPortalFromThrustTransposed : public ArrayPortalFromThrustBase
{
public:
  typedef T ValueType;
  typedef typename T::ComponentType ComponentType;
  static const vtkm::IdComponent NUM_COMPONENTS = ValueType::NUM_COMPONENTS;

  typedef typename thrust::system::cuda::pointer< ComponentType > PointerType;

  //none pointer type so that we use the lookup functor based wrapper
  typedef T IteratorType;

  VTKM_EXEC_CONT_EXPORT ArrayPortalFromThrustTransposed() {  }

  VTKM_CONT_EXPORT
  ArrayPortalFromThrustTransposed(vtkm::Vec<PointerType,NUM_COMPONENTS> begin,
                                  vtkm::Id length)
    : BeginIterators( begin ),
      NumberOfValues( length  )
      {  }

  /// Copy constructor for any other ArrayPortalFromThrustTransposed with an iterator
  /// type that can be copied to this iterator type. This allows us to do any
  /// type casting that the iterators do (like the non-const to const cast).
  ///
  template<typename OtherT>
  VTKM_EXEC_CONT_EXPORT
  ArrayPortalFromThrustTransposed(const ArrayPortalFromThrustTransposed<OtherT> &src)
    : BeginIterators(src.BeginIterators),
      NumberOfValues(src.NumberOfValues)
  {  }

  template<typename OtherT>
  VTKM_EXEC_CONT_EXPORT
  ArrayPortalFromThrustTransposed<T> &operator=(
      const ArrayPortalFromThrustTransposed<OtherT> &src)
  {
    this->BeginIterators = src.BeginIterators;
    this->NumberOfValues = src.NumberOfValues;
    return *this;
  }

  VTKM_EXEC_CONT_EXPORT
  vtkm::Id GetNumberOfValues() const {
    return NumberOfValues;
  }

  VTKM_EXEC_EXPORT
  ValueType Get(vtkm::Id index) const {
    return vtkm::exec::cuda::internal::to_value_type<ValueType>::get( index, this->BeginIterators );
  }

  VTKM_EXEC_EXPORT
  void Set(vtkm::Id index, ValueType value) const {
    vtkm::exec::cuda::internal::to_value_type<ValueType>::set(index, value, this->BeginIterators );
  }

private:
  vtkm::Vec<PointerType,NUM_COMPONENTS> BeginIterators;
  vtkm::Id NumberOfValues;
};

template<typename T>
class ConstArrayPortalFromThrustTransposed : public ArrayPortalFromThrustBase
{
public:
  typedef T ValueType;
  typedef typename T::ComponentType ComponentType;
  static const vtkm::IdComponent NUM_COMPONENTS = ValueType::NUM_COMPONENTS;

  typedef typename thrust::system::cuda::pointer< const ComponentType > PointerType;

  //none pointer type so that we use the lookup functor based wrapper
  typedef T IteratorType;

  VTKM_EXEC_CONT_EXPORT ConstArrayPortalFromThrustTransposed() {  }

  VTKM_CONT_EXPORT
  ConstArrayPortalFromThrustTransposed(vtkm::Vec<PointerType,NUM_COMPONENTS> begin,
                                       vtkm::Id length)
    : BeginIterators( begin ),
      NumberOfValues( length )
      {  }

  /// Copy constructor for any other ConstArrayPortalFromThrustTransposed with an iterator
  /// type that can be copied to this iterator type. This allows us to do any
  /// type casting that the iterators do (like the non-const to const cast).
  ///
  template<typename OtherT>
  VTKM_EXEC_CONT_EXPORT
  ConstArrayPortalFromThrustTransposed(const ConstArrayPortalFromThrustTransposed<OtherT> &src)
    : BeginIterators(src.BeginIterators),
      NumberOfValues(src.NumberOfValues)
  {  }

  template<typename OtherT>
  VTKM_EXEC_CONT_EXPORT
  ConstArrayPortalFromThrustTransposed<T> &operator=(
      const ConstArrayPortalFromThrustTransposed<OtherT> &src)
  {
    this->BeginIterators = src.BeginIterators;
    this->NumberOfValues = src.NumberOfValues;
    return *this;
  }

  VTKM_EXEC_CONT_EXPORT
  vtkm::Id GetNumberOfValues() const {
    return this->NumberOfValues;
  }

  VTKM_EXEC_EXPORT
  ValueType Get(vtkm::Id index) const {
    return vtkm::exec::cuda::internal::to_value_type<ValueType>::get( index, this->BeginIterators );
  }

  VTKM_EXEC_EXPORT
  void Set(vtkm::Id index, ValueType value) const {
    vtkm::exec::cuda::internal::to_value_type<ValueType>::set( index, value, this->BeginIterators );
  }

private:
  vtkm::Vec<PointerType,NUM_COMPONENTS> BeginIterators;
  vtkm::Id NumberOfValues;
};

}
}
}
} // namespace vtkm::exec::cuda::internal


#endif //vtk_m_exec_cuda_internal_ArrayPortalFromThrust_h

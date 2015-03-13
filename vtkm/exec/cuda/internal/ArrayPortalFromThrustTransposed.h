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


/// This templated implementation of an ArrayPortal allows you to adapt a pair
/// of begin/end iterators to an ArrayPortal interface.
///
template<typename T>
class ArrayPortalFromThrustTransposed : public ArrayPortalFromThrustBase
{
public:
  typedef T ValueType;
  typedef typename T::ComponentType ComponentType;
  typedef typename thrust::system::cuda::pointer< ComponentType > PointerType;
  typedef T* IteratorType;

  VTKM_EXEC_CONT_EXPORT ArrayPortalFromThrustTransposed() {  }

  VTKM_CONT_EXPORT
  ArrayPortalFromThrustTransposed(vtkm::Vec<PointerType,3> begin,
                                  vtkm::Vec<PointerType,3> end)
    : BeginIterators( begin ),
      EndIterators( end  )
      {  }

  /// Copy constructor for any other ArrayPortalFromThrustTransposed with an iterator
  /// type that can be copied to this iterator type. This allows us to do any
  /// type casting that the iterators do (like the non-const to const cast).
  ///
  template<typename OtherT>
  VTKM_EXEC_CONT_EXPORT
  ArrayPortalFromThrustTransposed(const ArrayPortalFromThrustTransposed<OtherT> &src)
    : BeginIterators(src.BeginIterators),
      EndIterators(src.EndIterators)
  {  }

  template<typename OtherT>
  VTKM_EXEC_CONT_EXPORT
  ArrayPortalFromThrustTransposed<T> &operator=(
      const ArrayPortalFromThrustTransposed<OtherT> &src)
  {
    this->BeginIterators = src.BeginIterators;
    this->EndIterators = src.EndIterators;
    return *this;
  }

  VTKM_EXEC_CONT_EXPORT
  vtkm::Id GetNumberOfValues() const {
    // Not using std::distance because on CUDA it cannot be used on a device.
    return (this->EndIterators[0] - this->BeginIterators[0]);
  }

  VTKM_EXEC_EXPORT
  ValueType Get(vtkm::Id index) const {
    return ValueType( *(this->BeginIterators[0]+index),
                      *(this->BeginIterators[1]+index),
                      *(this->BeginIterators[2]+index) );
  }

  VTKM_EXEC_EXPORT
  void Set(vtkm::Id index, ValueType value) const {
    *(this->BeginIterators[0]+index) = value[0];
    *(this->BeginIterators[1]+index) = value[1];
    *(this->BeginIterators[2]+index) = value[2];
  }

  // VTKM_CONT_EXPORT
  // IteratorType GetIteratorBegin() const { return this->BeginIterator.get(); }

  // VTKM_CONT_EXPORT
  // IteratorType GetIteratorEnd() const { return this->EndIterator.get(); }

private:
  vtkm::Vec<PointerType,3> BeginIterators;
  vtkm::Vec<PointerType,3> EndIterators;

  // VTKM_EXEC_EXPORT
  // PointerType IteratorAt(vtkm::Id index) const {
  //   // Not using std::advance because on CUDA it cannot be used on a device.
  //   return (this->BeginIterator + index);
  // }
};

template<typename T>
class ConstArrayPortalFromThrustTransposed : public ArrayPortalFromThrustBase
{
public:
  typedef T ValueType;
  typedef typename T::ComponentType ComponentType;
  typedef typename thrust::system::cuda::pointer< const ComponentType > PointerType;
  typedef const T* IteratorType;

  VTKM_EXEC_CONT_EXPORT ConstArrayPortalFromThrustTransposed() {  }

  VTKM_CONT_EXPORT
  ConstArrayPortalFromThrustTransposed(vtkm::Vec<PointerType,3> begin,
                                       vtkm::Vec<PointerType,3> end)
    : BeginIterators( begin ),
      EndIterators( end )
      {  }

  /// Copy constructor for any other ConstArrayPortalFromThrustTransposed with an iterator
  /// type that can be copied to this iterator type. This allows us to do any
  /// type casting that the iterators do (like the non-const to const cast).
  ///
  template<typename OtherT>
  VTKM_EXEC_CONT_EXPORT
  ConstArrayPortalFromThrustTransposed(const ConstArrayPortalFromThrustTransposed<OtherT> &src)
    : BeginIterators(src.BeginIterators),
      EndIterators(src.EndIterators)
  {  }

  template<typename OtherT>
  VTKM_EXEC_CONT_EXPORT
  ConstArrayPortalFromThrustTransposed<T> &operator=(
      const ConstArrayPortalFromThrustTransposed<OtherT> &src)
  {
    this->BeginIterators = src.BeginIterators;
    this->EndIterators = src.EndIterators;
    return *this;
  }

  VTKM_EXEC_CONT_EXPORT
  vtkm::Id GetNumberOfValues() const {
    // Not using std::distance because on CUDA it cannot be used on a device.
    return (this->EndIterators - this->BeginIterators);
  }

  VTKM_EXEC_EXPORT
  ValueType Get(vtkm::Id index) const {
    return ValueType(
              vtkm::exec::cuda::internal::load_through_texture< ComponentType >::get( (this->BeginIterators[0]+index) ),
              vtkm::exec::cuda::internal::load_through_texture< ComponentType >::get( (this->BeginIterators[1]+index) ),
              vtkm::exec::cuda::internal::load_through_texture< ComponentType >::get( (this->BeginIterators[2]+index) )
              );
  }

  VTKM_EXEC_EXPORT
  void Set(vtkm::Id index, ValueType value) const {
    *(this->BeginIterators[0] + index) = value[0];
    *(this->BeginIterators[1] + index) = value[1];
    *(this->BeginIterators[2] + index) = value[2];
  }

  // VTKM_CONT_EXPORT
  // IteratorType GetIteratorBegin() const { return this->BeginIterators.get(); }

  // VTKM_CONT_EXPORT
  // IteratorType GetIteratorEnd() const { return this->EndIterators.get(); }

private:
  vtkm::Vec<PointerType,3> BeginIterators;
  vtkm::Vec<PointerType,3> EndIterators;

  // VTKM_EXEC_EXPORT
  // PointerType IteratorAt(vtkm::Id index) const {
  //   // Not using std::advance because on CUDA it cannot be used on a device.
  //   return (this->BeginIterators + index);
  // }
};

}
}
}
} // namespace vtkm::exec::cuda::internal


#endif //vtk_m_exec_cuda_internal_ArrayPortalFromThrust_h

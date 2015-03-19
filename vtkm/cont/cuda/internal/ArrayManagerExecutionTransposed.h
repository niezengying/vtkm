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
#ifndef vtk_m_cont_cuda_internal_ArrayManagerExecutionTransposed_h
#define vtk_m_cont_cuda_internal_ArrayManagerExecutionTransposed_h

#include <vtkm/cont/Storage.h>
#include <vtkm/cont/ErrorControlOutOfMemory.h>

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

#include <thrust/system/cuda/vector.h>
#include <thrust/copy.h>

#if defined(__GNUC__) && !defined(VTKM_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif // gcc version >= 4.6
#endif // gcc && !CUDA

#include <vtkm/exec/cuda/internal/ArrayPortalFromThrustTransposed.h>
#include <boost/utility/enable_if.hpp>


namespace vtkm {
namespace cont {
namespace cuda {
namespace internal {

/// \c ArrayManagerExecutionThrustDevice provides an implementation for a \c
/// ArrayManagerExecution class for a thrust device adapter that is designed
/// for the cuda backend which has separate memory spaces for host and device. This
/// implementation contains a ::thrust::system::cuda::vector to allocate and manage
/// the array.
///
/// This array manager should only be used with the cuda device adapter,
/// since in the future it will take advantage of texture memory and
/// the unique memory access patterns of cuda systems.


//all arrays of complex type ( not in types.h of more than length 1).
//all arrays of length 3, 5, 6, 7, 8, 9 and up of float32, float64, int32, uint32, int64, uint64, etc
//hmmmmm do we use some sizeof tricks?
//do we first try all arrays of size 2 and up get transposed and see how perf is affected?
//I think sizeof combined with num elements


//by default all get transposed
template<typename T, int Size> struct TransposeToStructOfArrays   {typedef  boost::true_type type;};

//in-valid types
template<typename T> struct TransposeToStructOfArrays< T, 1 >   {typedef boost::false_type type; };
template<typename T> struct TransposeToStructOfArrays< T, 2 >   {typedef boost::false_type type; };

template<typename T, int NumComponents,  class StorageTag>
class ArrayManagerExecutionThrustDevice< vtkm::Vec< T, NumComponents >,
                                         StorageTag,
                                         typename ::boost::enable_if< typename TransposeToStructOfArrays<T, NumComponents>::type >::type
                                         >
{
public:
  typedef vtkm::Vec< T, NumComponents > ValueType;

  typedef vtkm::cont::internal::Storage<ValueType, StorageTag> ContainerType;

  typedef vtkm::exec::cuda::internal::ArrayPortalFromThrustTransposed< ValueType > PortalType;

  //this is non const as we return a value, not pointer when we are transposed
  typedef vtkm::exec::cuda::internal::ConstArrayPortalFromThrustTransposed< ValueType > PortalConstType;

  VTKM_CONT_EXPORT ArrayManagerExecutionThrustDevice():
    Arrays()
  {
  }

  ~ArrayManagerExecutionThrustDevice()
  {
    this->ReleaseResources();
  }

  /// Returns the size of the array.
  ///
  VTKM_CONT_EXPORT vtkm::Id GetNumberOfValues() const {
    return this->Arrays[0].size();
  }

  /// Allocates the appropriate size of the array and copies the given data
  /// into the array.
  ///
  template<class PortalControl>
  VTKM_CONT_EXPORT void LoadDataForInput(PortalControl arrayPortal)
  {
    //don't bind to the texture yet, as we could have allocate the array
    //on a previous call with AllocateArrayForOutput and now are directly
    //calling get portal const
    try
      {
      const vtkm::Id length = arrayPortal.GetNumberOfValues();
      const T* source = reinterpret_cast< const T* >(arrayPortal.GetRawIterator());

      #pragma unroll
      for(int i=0; i < NumComponents; ++i)
        {
        this->Arrays[i].reserve( length );

        //offset based on the position of the first element. e.g.
        //for x,y,z we need to point to y0 not x0 for i==1
        cudaMemcpy2D(this->Arrays[i].data().get(),
                     sizeof(T),
                     source + i,
                     NumComponents*sizeof(T),
                     sizeof(T),
                     length,
                     cudaMemcpyHostToDevice);
        }
      }
    catch (std::bad_alloc error)
      {
      throw vtkm::cont::ErrorControlOutOfMemory(error.what());
      }
  }

  /// Allocates the appropriate size of the array and copies the given data
  /// into the array.
  ///
  template<class PortalControl>
  VTKM_CONT_EXPORT void LoadDataForInPlace(PortalControl arrayPortal)
  {
    this->LoadDataForInput(arrayPortal);
  }

  /// Allocates the array to the given size.
  ///
  VTKM_CONT_EXPORT void AllocateArrayForOutput(
      ContainerType &vtkmNotUsed(container),
      vtkm::Id numberOfValues)
  {
    try
      {
      #pragma unroll
      for(int i=0; i < NumComponents; ++i)
        {
        this->Arrays[i].resize( numberOfValues );
        }
      }
    catch (std::bad_alloc error)
      {
      throw vtkm::cont::ErrorControlOutOfMemory(error.what());
      }


  }

  /// Allocates enough space in \c controlArray and copies the data in the
  /// device vector into it.
  ///
  VTKM_CONT_EXPORT void RetrieveOutputData(ContainerType &controlArray) const
  {
    const vtkm::Id length = this->Arrays[0].size();
    controlArray.Allocate(length);

    T* dest = reinterpret_cast< T* >(controlArray.GetPortal().GetRawIterator());

    #pragma unroll
    for(int i=0; i < NumComponents; ++i)
        {
        //offset based on the position of the first element. e.g.
        //for x,y,z we need to point to y0 not x0 for i==1
        cudaMemcpy2D(dest+i,
                     NumComponents*sizeof(T),
                     this->Arrays[i].data().get(),
                     sizeof(T),
                     sizeof(T),
                     length,
                     cudaMemcpyDeviceToHost);
        }
  }

  /// Resizes the device vector.
  ///
  VTKM_CONT_EXPORT void Shrink(vtkm::Id numberOfValues)
  {
    // The operation will succeed even if this assertion fails, but this
    // is still supposed to be a precondition to Shrink.
    VTKM_ASSERT_CONT(numberOfValues <= this->Arrays[0].size());

    #pragma unroll
    for(int i=0; i < NumComponents; ++i)
      {
      this->Arrays[i].resize( numberOfValues );
      }
  }

  VTKM_CONT_EXPORT PortalType GetPortal()
  {
    const std::size_t len = this->Arrays[0].size();
    vtkm::Vec< thrust::system::cuda::pointer< T >, NumComponents > v;

    #pragma unroll
    for(int i=0; i < NumComponents; ++i)
      {
      v[i] = this->Arrays[i].data();
      }
    return PortalType(v, len);

  }

  VTKM_CONT_EXPORT PortalConstType GetPortalConst() const
  {
    const std::size_t len = this->Arrays[0].size();
    vtkm::Vec< thrust::system::cuda::pointer< const T >, NumComponents > v;

    #pragma unroll
    for(int i=0; i < NumComponents; ++i)
      {
      v[i] = this->Arrays[i].data();
      }
    return PortalConstType( v, len);
  }


  /// Frees all memory.
  ///
  VTKM_CONT_EXPORT void ReleaseResources()
  {

    #pragma unroll
    for(int i=0; i < NumComponents; ++i)
      {
      this->Arrays[i].clear();
      this->Arrays[i].shrink_to_fit();
      }
  }

private:
  // Not implemented
  ArrayManagerExecutionThrustDevice(
      ArrayManagerExecutionThrustDevice<ValueType, StorageTag> &);
  void operator=(
      ArrayManagerExecutionThrustDevice<ValueType, StorageTag> &);

  ::thrust::system::cuda::vector< T > Arrays[NumComponents];
};


}
}
}
} // namespace vtkm::cont::cuda::internal

#endif // vtk_m_cont_cuda_internal_ArrayManagerExecutionThrustDevice_h

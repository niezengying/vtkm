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
template<class StorageTag>
class ArrayManagerExecutionThrustDevice< vtkm::Vec< vtkm::Float32, 3 >, StorageTag>
{
public:
  typedef vtkm::Vec< vtkm::Float32, 3 > ValueType;

  typedef vtkm::cont::internal::Storage<ValueType, StorageTag> ContainerType;

  typedef vtkm::exec::cuda::internal::ArrayPortalFromThrustTransposed< ValueType > PortalType;
  typedef vtkm::exec::cuda::internal::ConstArrayPortalFromThrustTransposed< const ValueType > PortalConstType;

  VTKM_CONT_EXPORT ArrayManagerExecutionThrustDevice():
    ArrayX(),
    ArrayY(),
    ArrayZ()
  {
  }

  ~ArrayManagerExecutionThrustDevice()
  {
    this->ReleaseResources();
  }

  /// Returns the size of the array.
  ///
  VTKM_CONT_EXPORT vtkm::Id GetNumberOfValues() const {
    return this->ArrayX.size();
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
      this->ArrayX.reserve( length );
      this->ArrayY.reserve( length );
      this->ArrayZ.reserve( length );

      cudaMemcpy2D(this->ArrayX.data().get(),
                   sizeof(vtkm::Float32),
                   arrayPortal.GetRawIterator(),
                   3*sizeof(vtkm::Float32),
                   sizeof(vtkm::Float32),
                   length,
                   cudaMemcpyHostToDevice);

      cudaMemcpy2D(this->ArrayY.data().get(),
                   sizeof(vtkm::Float32),
                   arrayPortal.GetRawIterator()+1,
                   3*sizeof(vtkm::Float32),
                   sizeof(vtkm::Float32),
                   length,
                   cudaMemcpyHostToDevice);

      cudaMemcpy2D(this->ArrayZ.data().get(),
                   sizeof(vtkm::Float32),
                   arrayPortal.GetRawIterator()+2,
                   3*sizeof(vtkm::Float32),
                   sizeof(vtkm::Float32),
                   length,
                   cudaMemcpyHostToDevice);
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
      this->ArrayX.resize(numberOfValues);
      this->ArrayY.resize(numberOfValues);
      this->ArrayZ.resize(numberOfValues);
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
    const vtkm::Id length = this->ArrayX.size();
    controlArray.Allocate(length);


    cudaMemcpy2D(controlArray.GetPortal().GetRawIterator(),
                 3*sizeof(vtkm::Float32),
                 this->ArrayX.data().get(),
                 sizeof(vtkm::Float32),
                 sizeof(vtkm::Float32),
                 length,
                 cudaMemcpyDeviceToHost);

    cudaMemcpy2D(controlArray.GetPortal().GetRawIterator(),
                 3*sizeof(vtkm::Float32),
                 this->ArrayY.data().get(),
                 sizeof(vtkm::Float32),
                 sizeof(vtkm::Float32),
                 length,
                 cudaMemcpyDeviceToHost);

    cudaMemcpy2D(controlArray.GetPortal().GetRawIterator(),
                 3*sizeof(vtkm::Float32),
                 this->ArrayZ.data().get(),
                 sizeof(vtkm::Float32),
                 sizeof(vtkm::Float32),
                 length,
                 cudaMemcpyDeviceToHost);
  }

  /// Resizes the device vector.
  ///
  VTKM_CONT_EXPORT void Shrink(vtkm::Id numberOfValues)
  {
    // The operation will succeed even if this assertion fails, but this
    // is still supposed to be a precondition to Shrink.
    VTKM_ASSERT_CONT(numberOfValues <= this->ArrayX.size());

    this->ArrayX.resize(numberOfValues);
    this->ArrayY.resize(numberOfValues);
    this->ArrayZ.resize(numberOfValues);
  }

  VTKM_CONT_EXPORT PortalType GetPortal()
  {
    const std::size_t len = this->ArrayX.size();
    return PortalType( vtkm::make_Vec(this->ArrayX.data(),
                                      this->ArrayY.data(),
                                      this->ArrayZ.data() ),
                       vtkm::make_Vec(this->ArrayX.data() + len,
                                      this->ArrayY.data() + len,
                                      this->ArrayZ.data() + len )
                       );
  }

  VTKM_CONT_EXPORT PortalConstType GetPortalConst() const
  {
    const std::size_t len = this->ArrayX.size();
    return PortalConstType( vtkm::make_Vec(this->ArrayX.data(),
                                      this->ArrayY.data(),
                                      this->ArrayZ.data() ),
                            vtkm::make_Vec(this->ArrayX.data() + len,
                                      this->ArrayY.data() + len,
                                      this->ArrayZ.data() + len )
                            );
  }


  /// Frees all memory.
  ///
  VTKM_CONT_EXPORT void ReleaseResources()
  {
    this->ArrayX.clear();
    this->ArrayY.clear();
    this->ArrayZ.clear();
    this->ArrayX.shrink_to_fit();
    this->ArrayY.shrink_to_fit();
    this->ArrayZ.shrink_to_fit();
  }

private:
  // Not implemented
  ArrayManagerExecutionThrustDevice(
      ArrayManagerExecutionThrustDevice<ValueType, StorageTag> &);
  void operator=(
      ArrayManagerExecutionThrustDevice<ValueType, StorageTag> &);


  ::thrust::system::cuda::vector< vtkm::Float32 > ArrayX;
  ::thrust::system::cuda::vector< vtkm::Float32 > ArrayY;
  ::thrust::system::cuda::vector< vtkm::Float32 > ArrayZ;
};


}
}
}
} // namespace vtkm::cont::cuda::internal

#endif // vtk_m_cont_cuda_internal_ArrayManagerExecutionThrustDevice_h

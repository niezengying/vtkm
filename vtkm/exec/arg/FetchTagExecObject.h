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
#ifndef vtk_m_exec_arg_FetchTagExecObject_h
#define vtk_m_exec_arg_FetchTagExecObject_h

#include <vtkm/exec/arg/AspectTagDefault.h>
#include <vtkm/exec/arg/Fetch.h>

#include <vtkm/exec/ExecutionObjectBase.h>

#include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace vtkm {
namespace exec {
namespace arg {

/// \brief \c Fetch tag for execution objects.
///
/// \c FetchTagExecObject is a tag used with the \c Fetch class to retreive
/// execution objects. For safety, execution objects are read-only. \c
/// FetchTagExecObject is almost always used in conjunction with \c
/// TransportTagExecObject and vice versa.
///
struct FetchTagExecObject {  };


template<typename Invocation, vtkm::IdComponent ParameterIndex>
struct Fetch<
    vtkm::exec::arg::FetchTagExecObject,
    vtkm::exec::arg::AspectTagDefault,
    Invocation,
    ParameterIndex>
{
  typedef typename Invocation::ParameterInterface::
      template ParameterType<ParameterIndex>::type ExecObjectType;

  // If you get a compile error here, it means you tried to use an object that
  // is not an execution object as an argument that is expected to be one. All
  // execution objects are expected to inherit from
  // vtkm::exec::ExecutionObjectBase.
  BOOST_MPL_ASSERT(( boost::is_base_of<vtkm::exec::ExecutionObjectBase, ExecObjectType> ));

  typedef ExecObjectType ValueType;

  VTKM_EXEC_EXPORT
  ValueType Load(vtkm::Id vtkmNotUsed(index),
                 const Invocation &invocation) const
  {
    return invocation.Parameters.template GetParameter<ParameterIndex>();
  }

  VTKM_EXEC_EXPORT
  void Store(vtkm::Id, const Invocation &, const ValueType &) const
  {
    // Store is a no-op for this fetch.
  }
};

}
}
} // namespace vtkm::exec::arg

#endif //vtk_m_exec_arg_FetchTagExecObject_h

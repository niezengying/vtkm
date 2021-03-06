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
// **** DO NOT EDIT THIS FILE!!! ****
// This file is automatically generated by FunctionInterfaceDetailPre.h.in

#if !defined(vtk_m_worklet_internal_DispatcherBase_h)
#error DispatcherBaseDetailInvoke.h must be included from DispatcherBase.h
#endif

// Note that this file is included from the middle of the DispatcherBase.h
// class to provide the implementation of the Invoke method, which requires
// variable length template args. If we choose to support C++11 variable
// template arguments, then this could all be replaced efficiently with a
// single method with a variadic template function that calls
// make_FunctionInterface.



template<typename T1>
VTKM_CONT_EXPORT
void Invoke(T1 a1) const
{
  this->StartInvoke(
        vtkm::internal::make_FunctionInterface<void>(a1));
}

template<typename T1,
         typename T2>
VTKM_CONT_EXPORT
void Invoke(T1 a1, T2 a2) const
{
  this->StartInvoke(
        vtkm::internal::make_FunctionInterface<void>(a1,a2));
}

template<typename T1,
         typename T2,
         typename T3>
VTKM_CONT_EXPORT
void Invoke(T1 a1, T2 a2, T3 a3) const
{
  this->StartInvoke(
        vtkm::internal::make_FunctionInterface<void>(a1,a2,a3));
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4>
VTKM_CONT_EXPORT
void Invoke(T1 a1, T2 a2, T3 a3, T4 a4) const
{
  this->StartInvoke(
        vtkm::internal::make_FunctionInterface<void>(a1,a2,a3,a4));
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5>
VTKM_CONT_EXPORT
void Invoke(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5) const
{
  this->StartInvoke(
        vtkm::internal::make_FunctionInterface<void>(a1,a2,a3,a4,a5));
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6>
VTKM_CONT_EXPORT
void Invoke(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6) const
{
  this->StartInvoke(
        vtkm::internal::make_FunctionInterface<void>(a1,a2,a3,a4,a5,a6));
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename T7>
VTKM_CONT_EXPORT
void Invoke(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6, T7 a7) const
{
  this->StartInvoke(
        vtkm::internal::make_FunctionInterface<void>(a1,a2,a3,a4,a5,a6,a7));
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename T7,
         typename T8>
VTKM_CONT_EXPORT
void Invoke(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6, T7 a7, T8 a8) const
{
  this->StartInvoke(
        vtkm::internal::make_FunctionInterface<void>(a1,a2,a3,a4,a5,a6,a7,a8));
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename T7,
         typename T8,
         typename T9>
VTKM_CONT_EXPORT
void Invoke(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6, T7 a7, T8 a8, T9 a9) const
{
  this->StartInvoke(
        vtkm::internal::make_FunctionInterface<void>(a1,a2,a3,a4,a5,a6,a7,a8,a9));
}


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

#include <vtkm/worklet/internal/DispatcherBase.h>

#include <vtkm/cont/DeviceAdapterSerial.h>

#include <vtkm/cont/arg/ControlSignatureTagBase.h>

#include <vtkm/worklet/internal/WorkletBase.h>

#include <vtkm/cont/testing/Testing.h>

namespace {

typedef vtkm::cont::DeviceAdapterTagSerial Device;

static const vtkm::Id ARRAY_SIZE = 10;

struct TestExecObject
{
  VTKM_EXEC_CONT_EXPORT
  TestExecObject() : Array(NULL) {  }

  VTKM_EXEC_CONT_EXPORT
  TestExecObject(vtkm::Id *array) : Array(array) {  }

  vtkm::Id *Array;
};

struct TestTypeCheckTag {  };
struct TestTransportTag {  };
struct TestFetchTagInput {  };
struct TestFetchTagOutput {  };

} // anonymous namespace

namespace vtkm {
namespace cont {
namespace arg {

template<>
struct TypeCheck<TestTypeCheckTag, vtkm::Id *>
{
  static const bool value = true;
};

template<>
struct Transport<TestTransportTag, vtkm::Id *, Device>
{
  typedef TestExecObject ExecObjectType;

  VTKM_CONT_EXPORT
  ExecObjectType operator()(vtkm::Id *contData, vtkm::Id size) const
  {
    VTKM_TEST_ASSERT(size == ARRAY_SIZE,
                     "Got unexpected size in test transport.");
    return ExecObjectType(contData);
  }
};

}
}
} // namespace vtkm::cont::arg

namespace vtkm {
namespace exec {
namespace arg {

template<typename Invocation, vtkm::IdComponent ParameterIndex>
struct Fetch<TestFetchTagInput, vtkm::exec::arg::AspectTagDefault, Invocation, ParameterIndex>
{
  typedef vtkm::Id ValueType;

  VTKM_EXEC_EXPORT
  ValueType Load(vtkm::Id index, const Invocation &invocation) const {
    return invocation.Parameters.
        template GetParameter<ParameterIndex>().Array[index];
  }

  VTKM_EXEC_EXPORT
  void Store(vtkm::Id, const Invocation &, ValueType) const {
    // No-op
  }
};

template<typename Invocation, vtkm::IdComponent ParameterIndex>
struct Fetch<TestFetchTagOutput, vtkm::exec::arg::AspectTagDefault, Invocation, ParameterIndex>
{
  typedef vtkm::Id ValueType;

  VTKM_EXEC_EXPORT
  ValueType Load(vtkm::Id, const Invocation &) const {
    // No-op
    return ValueType();
  }

  VTKM_EXEC_EXPORT
  void Store(vtkm::Id index,
             const Invocation &invocation,
             ValueType value) const {
    invocation.Parameters.template GetParameter<ParameterIndex>().Array[index] =
        value;
  }
};

}
}
} // vtkm::exec::arg

namespace {

class TestWorkletBase : public vtkm::worklet::internal::WorkletBase
{
public:
  struct TestIn : vtkm::cont::arg::ControlSignatureTagBase {
    typedef TestTypeCheckTag TypeCheckTag;
    typedef TestTransportTag TransportTag;
    typedef TestFetchTagInput FetchTag;
  };
  struct TestOut : vtkm::cont::arg::ControlSignatureTagBase {
    typedef TestTypeCheckTag TypeCheckTag;
    typedef TestTransportTag TransportTag;
    typedef TestFetchTagOutput FetchTag;
  };
};

class TestWorklet : public TestWorkletBase
{
public:
  typedef void ControlSignature(TestIn, TestOut);
  typedef _2 ExecutionSignature(_1, WorkIndex);

  VTKM_EXEC_EXPORT
  vtkm::Id operator()(vtkm::Id value, vtkm::Id index) const
  {
    VTKM_TEST_ASSERT(value == TestValue(index, vtkm::Id()),
                     "Got bad value in worklet.");
    return TestValue(index, vtkm::Id()) + 1000;
  }
};

#define ERROR_MESSAGE "Expected worklet error."

class TestErrorWorklet : public TestWorkletBase
{
public:
  typedef void ControlSignature(TestIn, TestOut);
  typedef void ExecutionSignature(_1, _2);

  VTKM_EXEC_EXPORT
  void operator()(vtkm::Id, vtkm::Id) const
  {
    this->RaiseError(ERROR_MESSAGE);
  }
};

template<typename WorkletType>
class TestDispatcher :
    public vtkm::worklet::internal::DispatcherBase<
      TestDispatcher<WorkletType>,
      WorkletType,
      TestWorkletBase,
      Device>
{
  typedef vtkm::worklet::internal::DispatcherBase<
      TestDispatcher<WorkletType>,
      WorkletType,
      TestWorkletBase,
      Device> Superclass;
  typedef vtkm::internal::FunctionInterface<void(vtkm::Id *, vtkm::Id *)>
      ParameterInterface;
  typedef vtkm::internal::Invocation<
      ParameterInterface,
      typename Superclass::ControlInterface,
      typename Superclass::ExecutionInterface,
      1> Invocation;
public:
  VTKM_CONT_EXPORT
  TestDispatcher(const WorkletType &worklet = WorkletType())
    : Superclass(worklet) {  }

  VTKM_CONT_EXPORT
  void DoInvoke(const Invocation &invocation) const
  {
    std::cout << "In TestDispatcher::DoInvoke()" << std::endl;
    this->BasicInvoke(invocation, ARRAY_SIZE);
  }

private:
  WorkletType Worklet;
};

void TestBasicInvoke()
{
  std::cout << "Test basic invoke" << std::endl;
  std::cout << "  Set up data." << std::endl;
  vtkm::Id inputArray[ARRAY_SIZE];
  vtkm::Id outputArray[ARRAY_SIZE];

  for (vtkm::Id index = 0; index < ARRAY_SIZE; index++)
  {
    inputArray[index] = TestValue(index, vtkm::Id());
    outputArray[index] = 0xDEADDEAD;
  }

  std::cout << "  Create and run dispatcher." << std::endl;
  TestDispatcher<TestWorklet> dispatcher;
  dispatcher.Invoke(inputArray, outputArray);

  std::cout << "  Check output of invoke." << std::endl;
  for (vtkm::Id index = 0; index < ARRAY_SIZE; index++)
  {
    VTKM_TEST_ASSERT(outputArray[index] == TestValue(index, vtkm::Id()) + 1000,
                     "Got bad value from testing.");
  }
}

void TestInvokeWithError()
{
  std::cout << "Test invoke with error raised" << std::endl;
  std::cout << "  Set up data." << std::endl;
  vtkm::Id inputArray[ARRAY_SIZE];
  vtkm::Id outputArray[ARRAY_SIZE];

  for (vtkm::Id index = 0; index < ARRAY_SIZE; index++)
  {
    inputArray[index] = TestValue(index, vtkm::Id());
    outputArray[index] = 0xDEADDEAD;
  }

  try
  {
    std::cout << "  Create and run dispatcher that raises error." << std::endl;
    TestDispatcher<TestErrorWorklet> dispatcher;
    dispatcher.Invoke(inputArray, outputArray);
    VTKM_TEST_FAIL("Exception not thrown.");
  }
  catch (vtkm::cont::ErrorExecution error)
  {
    std::cout << "  Got expected exception." << std::endl;
    VTKM_TEST_ASSERT(error.GetMessage() == ERROR_MESSAGE,
                     "Got unexpected error message.");
  }
}

void TestInvokeWithBadType()
{
  std::cout << "Test invoke with bad type" << std::endl;

  vtkm::Id array[ARRAY_SIZE];
  TestDispatcher<TestWorklet> dispatcher;

  try
  {
    std::cout << "  First argument bad." << std::endl;
    dispatcher.Invoke(NULL, array);
  }
  catch (vtkm::cont::ErrorControlBadType error)
  {
    std::cout << "    Got expected exception." << std::endl;
    std::cout << "    " << error.GetMessage() << std::endl;
    VTKM_TEST_ASSERT(error.GetMessage().find(" 1 ") != std::string::npos,
                     "Parameter index not named in error message.");
  }

  try
  {
    std::cout << "  Second argument bad." << std::endl;
    dispatcher.Invoke(array, NULL);
  }
  catch (vtkm::cont::ErrorControlBadType error)
  {
    std::cout << "    Got expected exception." << std::endl;
    std::cout << "    " << error.GetMessage() << std::endl;
    VTKM_TEST_ASSERT(error.GetMessage().find(" 2 ") != std::string::npos,
                     "Parameter index not named in error message.");
  }
}

void TestDispatcherBase()
{
  TestBasicInvoke();
  TestInvokeWithError();
  TestInvokeWithBadType();
}

} // anonymous namespace

int UnitTestDispatcherBase(int, char *[])
{
  return vtkm::cont::testing::Testing::Run(TestDispatcherBase);
}

#define PY_SSIZE_T_CLEAN
#define ARG_OFFSET 5

#include <Python.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <iostream>

// #define USE_NVTX
#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif

//Meta-data format we will use
#include <THCTensorInfo.cuh>

//Cuda kernels
#include <kernel.h>

#define ERROR_MSG cout << "Error at " << __FILE__ << ":" << __LINE__ << "\n";

using namespace std;
TensorInfo<void, IDXTYPE> PyOb_2_tinfo(PyObject* tensor, float_types data_type)
{
  PyObject* PyStrides = PyObject_CallMethod(tensor, "stride", NULL);
  if(PyStrides == NULL)
  {
    ERROR_MSG;
    cout << "PyStrides = NULL" << endl;
  }

  PyObject* PySizes = PyObject_CallMethod(tensor, "size", NULL);
  if(PySizes == NULL)
  {
    ERROR_MSG;
    cout << "PySizes = NULL" << endl;
  }

  PyObject* PyDataPtr = PyObject_CallMethod(tensor, "data_ptr", NULL);
  if(PyDataPtr == NULL)
  {
    ERROR_MSG;
    cout << "PyDataPtr = NULL" << endl;
  }

  void* data_ptr = (void*) PyLong_AsLong(PyDataPtr);
  Py_ssize_t ndims = PyList_GET_SIZE(PySizes);
  //TODO put proper checking on ndims < MAX_CUTORCH_DIMS
  IDXTYPE strides[MAX_CUTORCH_DIMS], sizes[MAX_CUTORCH_DIMS];

  for(int i = 0; i < ndims; i++)
  {
    strides[i] = PyLong_AsLong(PyTuple_GetItem(PyStrides, i));
    sizes[i] = PyLong_AsLong(PyTuple_GetItem(PySizes, i));
  }

  // Reference counts still behave strangely, but at least these appear to cap 
  // the process' memory usage.
  Py_DECREF(PyStrides);
  Py_DECREF(PySizes);
  Py_DECREF(PyDataPtr);

  return TensorInfo<void, IDXTYPE>(data_ptr, ndims, sizes, strides, data_type);
}

vector<TensorInfo<void, IDXTYPE> > get_TInfos(PyObject* args)
{
  vector<TensorInfo<void, IDXTYPE> > info_vec;
#ifdef DEBUG_ANY 
  cout << "Processing " << PyTuple_GET_SIZE(args) << " arguments" << endl;
#endif

#ifdef CHECK_MEMLEAK
  for(int iter = 0; iter < 1e7; iter++ )
#endif
    for(Py_ssize_t i=0; i<PyTuple_GET_SIZE(args); i++)
    {
      PyObject* pyTensor = PyTuple_GetItem(args, i);

      // check type, only take if Tensor, Variable, or Parameter
      string objType(pyTensor->ob_type->tp_name);

      PyObject* pyObjTypeCall = PyObject_CallMethod(pyTensor, "type", NULL);
      if(pyObjTypeCall == NULL)
      {
	ERROR_MSG;
	cout << "For args item " << i << ", pyObjTypeCall = NULL" << endl;
      }

      // This gives a segfault:
      // cout << "pyObjTypeCall direct conversion attempt = " << 
      //          PyBytes_AsString(pyObjTypeCall) << endl;

      PyObject* pyObjASCII = PyUnicode_AsASCIIString(pyObjTypeCall);
      if(pyObjASCII == NULL)
      {
	ERROR_MSG;
	cout << "For args item " << i << ", pyObjASCII = NULL " << endl;
      }

      // cout << "Py_REFCNT(pyObjTypeCall) = " << Py_REFCNT(pyObjTypeCall) << endl;
      Py_DECREF(pyObjTypeCall);

      string objTypeCall(PyBytes_AsString(pyObjASCII));

      // cout << "Py_REFCNT(pyObjASCII) = " << Py_REFCNT(pyObjASCII) << endl;
      Py_DECREF(pyObjASCII);

#ifdef DEBUG_ANY
      cout << "arg " << i << endl;
      cout << "objType = " << objType << endl;
      cout << "objTypeCall = " << objTypeCall << endl;
#endif

      if(objTypeCall == "torch.cuda.FloatTensor")
#ifdef CHECK_MEMLEAK
	if(iter == 0 )
#endif
	  info_vec.push_back(PyOb_2_tinfo(pyTensor, FLOAT));
#ifdef CHECK_MEMLEAK
	else
	  info_vec[i] = PyOb_2_tinfo(pyTensor, FLOAT);
#endif
      else if(objTypeCall == "torch.cuda.HalfTensor")
	info_vec.push_back(PyOb_2_tinfo(pyTensor, HALF));
      // Could add double
      else
      {
	ERROR_MSG;
	cout << "For args item " << i << ", unsupported .type() found: "
	     << objTypeCall << "\n"
		"Supported types:\n"
		"torch.cuda.FloatTensor\n"
		"torch.cuda.HalfTensor\n"
		"torch.autograd.variable.Variable containing FloatTensor\n"
		"torch.autograd.variable.Variable containing HalfTensor\n"
		"torch.nn.parameter.Parameter containing FloatTensor\n" 
		"torch.nn.parameter.Parameter containing HalfTensor\n" 
	     << endl;
      }
    }

  // PyErr_SetString(PyExc_RuntimeError, "Exception set in  ");

  return info_vec;
}

//Will extract all tensors in order. Assumes flat structure, tensors can not be wrapped in lists
//tuples or any other iterator structure.
static PyObject* norm_fwd(PyObject* self, PyObject* args)
{
#ifdef USE_NVTX
nvtxRangePushA("norm_fwd C backend");
#endif

  vector<TensorInfo<void, IDXTYPE> >tensors = get_TInfos(args);

#ifdef DEBUG_ANY
  cout << "tensors.size() = " << tensors.size() << endl;
#endif

  IDXTYPE totalElems = 1;
  for(int i = 0; i < tensors[0].dims; i++ )
    totalElems *= tensors[0].sizes[i];
  send_to_fwd
  (
    tensors[0], 
    tensors[1], 
    tensors[2], 
    totalElems
  );

#ifdef USE_NVTX
nvtxRangePop();
#endif

  Py_RETURN_NONE;
}

static PyObject* norm_bwd(PyObject* self, PyObject* args)
{
#ifdef USE_NVTX
nvtxRangePushA("norm_bwd C backend");
#endif

  vector<TensorInfo<void, IDXTYPE> >tensors = get_TInfos(args);

#ifdef DEBUG_ANY
  cout << "tensors.size() = " << tensors.size() << endl;
#endif

  IDXTYPE totalElems = 1;
  for(int i = 0; i < tensors[0].dims; i++ )
    totalElems *= tensors[0].sizes[i];
  send_to_bwd
  (
    tensors[0], 
    tensors[1], 
    tensors[2], 
    tensors[3], 
    totalElems
  );

#ifdef USE_NVTX
nvtxRangePop();
#endif

  Py_RETURN_NONE;
}



//*******************PYTHON BOILER PLATE*******************
static PyMethodDef apex_methods[] = {
  {"norm_fwd", (PyCFunction) norm_fwd, METH_VARARGS, "Slowest-dim norm, forward pass."},
  {"norm_bwd", (PyCFunction) norm_bwd, METH_VARARGS, "Slowest-dim norm, backward pass."},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

//Module Definitions
static struct PyModuleDef apex = {
  PyModuleDef_HEAD_INIT, "apex._C", "Module to add CUDA extensions to Pytorch.", -1, apex_methods
};
//Initialization Function
PyMODINIT_FUNC PyInit__C(void){

  //Let's throw an error if we can't find pytorch.
  PyImport_ImportModule("torch");
  Py_Initialize();
  return PyModule_Create(&apex);
}
#else
PyMODINIT_FUNC initMODULE(void){
  //Let's throw an error if we can't find pytorch.
  PyImport_ImportModule("torch");
  (void) Py_InitModule3("apex._C", apex, "A PyTorch Extension.");
}

#endif
//*********************************************************


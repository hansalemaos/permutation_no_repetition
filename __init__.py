import os
import subprocess
import sys
import math
import numpy as np
from zipint import zipint, unzipint

def _dummyimport():
    import Cython


try:
    from .cythonpermutation import permutation
except Exception as e:
    cstring = r"""# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: overflowcheck.fold=False
# cython: embedsignature=False
# cython: embedsignature.format=c
# cython: cdivision=True
# cython: cdivision_warnings=False
# cython: cpow=True
# cython: c_api_binop_methods=True
# cython: profile=False
# cython: linetrace=False
# cython: infer_types=False
# cython: language_level=3
# cython: c_string_type=bytes
# cython: c_string_encoding=default
# cython: type_version_tag=True
# cython: unraisable_tracebacks=False
# cython: iterable_coroutine=True
# cython: annotation_typing=True
# cython: emit_code_comments=False
# cython: cpp_locals=True
cimport cython
import numpy as np
cimport numpy as np
import cython

cdef void swap(np.int32_t[:] arr, int a, int b):
    cdef int temp
    temp = arr[a]
    arr[a] = arr[b]
    arr[b] = temp

cpdef permutation(np.int32_t [:] arr, np.int32_t[:] outarr, int start, int end, np.int32_t[:] totalcounter,set[int] checkset):
    cdef int j
    cdef int conco
    cdef int i
    if start == end:
            conco=0
            for j in range(end + 1):
                conco = conco + (arr[j] << (j*4))
            if conco not in checkset:
                for j in range(end + 1):
                    outarr[totalcounter[0]]=arr[j]
                    totalcounter[0]+=1
                checkset.add(conco)

    else:
        for i in range(start, end + 1):
            swap(arr, i, start)
            permutation(arr, outarr, start + 1, end, totalcounter, checkset)
            swap(arr, i, start)

"""
    pyxfile = f"bgv.pyx"
    pyxfilesetup = f"bgvcompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
        """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'cythonpermutation', 'sources': ['bgv.pyx'], 'include_dirs': [\'"""
        + numpyincludefolder
        + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='cythonpermutation',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    from .cythonpermutation import permutation





def _intarray_permutations(inputarray, start=0, end=-1):
    if end == -1:
        end = len(inputarray) - 1
    outputarraysize = math.factorial(len(inputarray)) * len(inputarray)
    outputarray = np.zeros(outputarraysize, dtype=np.int32)
    indexcounter = np.zeros(1, dtype=np.int32)
    checkset = set()
    permutation(inputarray, outputarray, start, end, indexcounter, checkset)
    return outputarray[: int(np.max(indexcounter))].reshape((-1, len(inputarray)))


def _ndarray2dint(arr, start=0, end=-1):
    setcheck = {}
    lu = []
    f1 = zipint(arr)
    c_ = -1
    for f_ in zip(f1):
        if f_ not in setcheck:
            c_ = c_ + 1
            setcheck[f_] = c_
            lu.append(c_)

        else:
            lu.append(setcheck.get(f_))
    b = _intarray_permutations(np.array(lu, dtype=np.int32), start=start, end=end)

    condlist = []
    choicelist = []
    setcheck = {}
    c_ = -1
    for f_ in zip(f1):
        if f_ not in setcheck:
            c_ = c_ + 1
            setcheck[f_] = c_
            condlist.append(b == c_)
            choicelist.append(f_)

    c = np.select(condlist, choicelist, 0)
    dt2 = np.dtype(
        object,
        metadata={
            "origshape": (c.shape[0], c.shape[1]),
            "zfill": f1.dtype.metadata["zfill"],
            "zfilltotal": f1.dtype.metadata["zfilltotal"],
        },
    )
    d = unzipint(np.array(c.ravel(), dtype=dt2))
    return d.reshape(
        (
            d.shape[0] // arr.shape[1],
            d.shape[1],
            arr.shape[1],
        )
    )


def _other_datatypes(a, start=0, end=-1):
    d1 = {}
    co = 0
    substlist = []
    for v in a:
        if v not in d1:
            d1[v] = co
            co += 1
        cav = d1.get(v)
        substlist.append(cav)
    b = _intarray_permutations(
        np.array(substlist, dtype=np.int32), start=start, end=end
    )
    condlist = []
    choicelist = []
    setcheck = set()
    for subs, original in zip(substlist, a):
        if subs in setcheck:
            continue
        condlist.append(b == subs)
        choicelist.append(original)
        setcheck.add(subs)
    return np.select(condlist, choicelist, a[0])


def cython_permutations(a, start=0, end=-1):
    if a.dtype == np.int32 and a.ndim == 1:
        return _intarray_permutations(a, start=start, end=end)

    if np.can_cast(a, np.int32) and a.ndim == 1:
        return _intarray_permutations(a.astype(np.int32), start=start, end=end)

    if a.ndim == 2:
        if np.dtype == np.int32:
            return _ndarray2dint(a, start=start, end=end)
        else:
            if np.can_cast(a, np.int32):
                return _ndarray2dint(a.astype(np.int32), start=start, end=end)
    if a.ndim == 1:
        return _other_datatypes(a, start=0, end=-1)
    raise NotImplementedError("array shape/dtype not implemented yet!")
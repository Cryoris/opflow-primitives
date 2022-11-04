import inspect
import sys
import setuptools

if not hasattr(setuptools, "find_namespace_packages") or not inspect.ismethod(
    setuptools.find_namespace_packages
):
    print(
        "Your setuptools version:'{}' does not support PEP 420 (find_namespace_packages). "
        "Upgrade it to version >='40.1.0' and repeat install.".format(
            setuptools.__version__
        )
    )
    sys.exit(1)

setuptools.setup(
    name="patchwork",
    version="0.1.0",
    description="Qiskit Opflow based Qiskit Primitives",
    author="Julien Gacon",
    author_email="julien.gacon@epfl.ch",
    license="Apache-2.0",
    classifiers=(
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ),
    keywords=["qiskit", "qiskit primitives"],
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
)

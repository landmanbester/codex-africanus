=======
History
=======

0.1.3 (2018-03-28)
------------------
* Fixed bug in im_to_vis. Added more tests for im_to_vis.
  Removed division by :math:`n` since it is trivial to reinstate
  after the fact. (:pr:`34`)
* Move numba implementations out of API functions. (:pr:`33`)
* Zernike Polynomial Direction Dependent Effects (:pr:`18`, :pr:`30`)
* Added division by :math:`n` to DFT.
  Fixed dask chunking issue.
  Updated test_vis_to_im_dask (:pr:`29`).
* Implement RIME visibility predict (:pr:`24`, :pr:`25`)
* Direct Fourier Transform (:pr:`19`)
* Parallactic Angle computation (:pr:`15`)
* Implement Feed Rotation term (:pr:`14`)
* Swap gridding correlation dimensions (:pr:`13`)
* Implement Direction Dependent Effect beam cubes (:pr:`12`)
* Implement Brightness Matrix Calculation (:pr:`9`)
* Implement RIME Phase Delay term (:pr:`8`)
* Support user supplied grids (:pr:`7`)
* Add dask wrappers to the gridder and degridder (:pr:`4`)
* Add weights to gridder/degridder and remove PSF function (:pr:`2`)

0.1.2 (2018-03-28)
------------------

* First release on PyPI.

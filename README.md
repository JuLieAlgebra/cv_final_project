# Final Project: Estimating Astronomical Distances via Photos

<!-- [![Release](https://img.shields.io/github/v/release/fpgmaas/cookiecutter-poetry-example)](https://img.shields.io/github/v/release/fpgmaas/cookiecutter-poetry-example) -->

<!-- [![Build status](https://github.com/JuLieAlgebra/photometric_redshift/actions/workflows/testing.yml/badge.svg?branch=master)](https://github.com/JuLieAlgebra/photometric_redshift/actions/workflows/testing.yml) -->

[![Maintainability](https://api.codeclimate.com/v1/badges/927273e266d47d404488/maintainability)](https://codeclimate.com/github/JuLieAlgebra/final_project/maintainability)

<!-- [![Test Coverage](https://api.codeclimate.com/v1/badges/927273e266d47d404488/test_coverage)](https://codeclimate.com/github/JuLieAlgebra/final_project/test_coverage) -->

<!-- [![Commit activity](https://img.shields.io/github/commit-activity/m/fpgmaas/cookiecutter-poetry-example)](https://img.shields.io/github/commit-activity/m/fpgmaas/cookiecutter-poetry-example)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://fpgmaas.github.io/cookiecutter-poetry-example/) -->
[![Code style with black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- [![Imports with isort](https://img.shields.io/badge/%20imports-isort-%231674b1)](https://pycqa.github.io/isort/) -->
<!-- [![License](https://img.shields.io/github/license/fpgmaas/cookiecutter-poetry-example)](https://img.shields.io/github/license/fpgmaas/cookiecutter-poetry-example)
 -->
Photometric redshift pipeline with, primarily, luigi and tensorflow. See docs link on the right for a more descriptive overview.

Note: For downloading the tabular data, I recommend adding a `.env` file with your AWS secrets like so:
``AWS_ACCESS_KEY_ID=something``
``AWS_SECRET_ACCESS_KEY=something``

The downloaded tabular data is 2.9MB.

However, you can also download the file here (preserve the file name or make sure that final_project/conf/aws_paths.yaml tabular_data entry matches the file name you chose): https://drive.google.com/file/d/1w1n8mihtWPwObA3JLvreH0BHdki3_y92/view?usp=sharing If you place the file into a folder called 'data' off the main directory of the repo, it will run without trying to download the file from AWS.

The SQL query used to grab the data via CasJobs:
```sql
SELECT TOP 10000
  za.specObjID, za.bestObjID, za.class, za.subClass, za.z, za.zErr,
  po.objID, po.type, po.flags, po.ra, po.dec,
  po.run, po.rerun, po.camcol, po.field,
  po.cModelFlux_u, po.cModelFlux_g, po.cModelFlux_r, po.cModelFlux_i, po.cModelFlux_z,
  (po.petroMag_r-po.extinction_r) as dered_petro_r,
  zp.z as zphot, zp.zErr as dzphot, zi.subclass as zsubclass,
  zi.e_bv_sfd,zi.primtarget, zi.sectarget,zi.targettype,zi.spectrotype
INTO MyDB.pasquet_2018_with_mag
FROM SpecObjAll as za
    JOIN PhotoObjAll as po ON (po.objID = za.bestObjID)
    JOIN Photoz as zp ON (zp.objID = za.bestObjID)
    JOIN galSpecInfo as zi ON (zi.SpecObjID = za.specObjID)
WHERE
    (za.z < 1 and za.z > 0 and za.zWarning=0)
    and (za.targetType ='SCIENCE' and za.survey='sdss')
    and (za.class='GALAXY' and zi.primtarget>=64)
    and (po.clean=1 and po.insideMask=0)
    and ((po.petroMag_r - po.extinction_r) <= 17.8)
ORDER BY RAND(10);
```

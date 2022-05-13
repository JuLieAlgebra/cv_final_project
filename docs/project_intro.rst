.. _project-intro:
*************
Project Intro
*************

I used Luigi and salted graphs for data extraction, preprocessing, and experimentation while I replicated (and modified) the smaller architecture from this paper: `Photometric redshifts from SDSS images using a Convolutional Neural Network <https://arxiv.org/abs/1806.06607>`_. I also reference this one quite a lot in my discussion: `Investigating Deep Learning Methods for Obtaining Photometric Redshift Estimations from Images <https://arxiv.org/abs/2109.02503v1>`_.

.. I'm replicating the smaller architecture from the first, and the idea of using a mixed-input model from the second.
The overall goal of the project is to produce an automated pipeline for training models to produce distance estimations of galaxies from photos in a learned approach. Below is a quick overview of the science behind why this is possible.

Problem Background
##################
How far away is this galaxy?

.. image:: images/galaxy_dist.png
  :width: 500
  :alt: Hubble Deep Field image with galaxy circled

In astronomy, until we develop light speed engines, we're almost entirely limited to estimating this by only cameras. It's a surprisingly hard task that's far from solved.

It also underpins every aspect of research. From determining the age of the universe and resolving crises in fundamental physics to being able to calibrate solar convection models, it impacts *everything*.


.. image:: images/problem_overview.png
  :width: 500
  :alt: Hubble Deep Field image with galaxy circled

Astronomers have developed a variety of methods, used for different distances and different situations, but we will talk about the most
accurate and its learned approximation.

Most astronomy pipelines today transform a cleaned image of an object in a particular filter to a single
number, magnitude, that can then be used in a machine learning approach. There are other things
calculated from these images that are used in other situations, but for redshift estimation, the work is
done with usually five or more magnitudes in different, non-overlapping filters.
Some astronomy groups have published papers skipping that transformation to train directly on the
images.

Instead of reducing information at the pixel level across all five filters to five single numbers,
they see a lot of potential in using that entire image of information to produce what usually five (good)
numbers prodce. Training directly on the images via CNNS often outperforms the tabular feature
extraction approach by some margin, but this method has struggled to be adopted, with the tradeoff
between the usually small performance boosts and the interpretability of the model. What is usually
done today are random forests or vanilla, smaller neural networks.

.. image:: images/doppler.png
  :width: 500


Definitions
############
(With hyperlinks for more information. Sorry for so much wikipedia, there weren’t a lot of good
general public explanations that don’t involve excessive jargon that I could find. Will find better
explanations to link to and better explain each idea in more detail for the actual report.)
`Redshift <https://en.wikipedia.org/wiki/Redshift>`_ – “distance” as measured by Doppler shift.
`Hubble Constant <https://lweb.cfa.harvard.edu/~dfabricant/huchra/hubble/>`_ – a time varying parameter that tells us the rate of expansion of the universe at that time epoch
`CCD <https://en.wikipedia.org/wiki/Charge-coupled_device>`_ – the camera of choice for most telescopes
`Spectroscopic Redshift <https://en.wikipedia.org/wiki/Redshift#Observations_in_astronomy>`_ – more accurate than photometric redshifts. Distance measurements produced by fitting spectra data to a black body curve
`Photometric Redshift <https://en.wikipedia.org/wiki/Photometric_redshift>`_ - distance measurements produced by photometric data
`Photometric <https://en.wikipedia.org/wiki/Photometry_(astronomy)>`_ – Data produced by photos of objects. Usually taken in a filter to only capture photos in a given range of wavelengths (red, blue, green, etc).
`SDSS <https://www.sdss.org/>`_ – Sloan Digital Sky Survey

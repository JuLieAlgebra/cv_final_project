.. _salted:
******
Salted
******

This is almost idential to Dr. Gorlin's `implementation <https://github.com/gorlins/salted>`_ in the salted demo. The only major deviation is that I added logic to avoid adding the partition luigi Parameters (ie, upper and lower) to the salt. That way, no matter how many workers are used for downloading images or pre-processing them, the work is not re-done on that basis alone.

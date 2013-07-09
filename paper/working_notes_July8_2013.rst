Notes from July 13 (flight)
===========================
Tried to re-approach problem by directly fitting H2CO tau values rather than
performing the simultaneous fit of two lines.  This approach has major advantages:

1. the lines are well-represented by gaussians and therefore the errors on tau
   are very well-recovered by the Levenberg-Marquardt algorithm (while the LVG
   fits were very poorly recovered).  The tau measurements can then be trivially
   represented as Normal variates in the MCMC sampler later on.
2. It's a lot faster.
3. It's easier to find mistakes.

Unfortunately, point #3 is important.  I discovered that I had been fitting
"tau" as the optical depth multiplier for the hyperfine lines assuming all
hyperfines have the same optical depth.  This is wrong.

*No it's not!!!*  I'm not assuming they have the same optical depth, I'm
assuming their optical depth is set such that the sum of the individual optical
depths divided by the total degeneracy is 1... that probably doesn't make sense
without an example.
For tau=1, you want the sum of the taus of the individual components to sum to 1.

Since RADEX / LVG models the H2CO lines as a single line, the modeled optical
depth is the *peak* optical depth for the whole complex *as if it were a
gaussian*.  Approximating the H2CO line as a Gaussian isn't terrible, though it
is important to account for the "wing" lines in order to achieve a good fit.
The problem is that the *peak* optical depth is ~1.3 times larger than the
individual hyperfine optical depths.

This is a deep error (i.e., it was fundamentally wrong in a huge number of
computations) and reflects some serious underlying stupidity on my part.  This
is exactly the kind of error I'm always terribly worried is hiding underneath
my work.  No one else is likely to catch it, ever.  They'll just disbelieve my
results if they disagree.

I've now corrected the error, and the results are significantly different.
Lucky for me, they go in the same direction I've been arguing for in the text.


What is the optical depth?
--------------------------
There are three possible interpretations:

1. The optical depth of one line (e.g., 1-1) is the sum of the optical depths of the individual components
2. The optical depth of one line is the peak of the sum of the total profile
3. The optical depth of one line is the optical depth of each of its hyperfine components

Only #1 makes physical sense.

Wait what?  Isn't that what I had before?

:math:`\tau_{1-1} = \sum_{hf} \tau_{hf}` 




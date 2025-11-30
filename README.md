# SymbolicRegressionWithGPU

Learning how to use symbolic regression using GPU's for calculations in python using some of the following sources.
- https://geppy.readthedocs.io/en/latest/intro_GEP.html
- https://github.com/ShuhuaGao/geppy/blob/master/examples/sr/v1.0.3_YourSymbolicRegression-Multidemic.ipynb
- https://blog.hpc.qmul.ac.uk/strategies-multi-node-gpu/#:~:text=when%20it%20finishes.-,multiprocessing,function%20but%20with%20different%20parameters.
- https://docs.cupy.dev/en/stable/overview.html
- https://deap.readthedocs.io/en/master/
- https://numba.pydata.org/
- https://www.sympy.org/en/index.html

Goal is to find the most optimal solution to the equation


Solve $F_{a},F_{A},F_{P},F_{d},F_{D},F_{h}$ Where

$$
\frac{⌊\frac{(\frac{2 \times A}{5}+2) \times P \times \frac{⌊\frac{2 \times a \times A}{100}⌋+5}{⌊\frac{2 \times d \times D}{100}⌋+5}}{50}+2⌉}{⌊\frac{2 \times h \times D}{100}⌋+D+10} \approx \frac{F_{a}(a,A)+F_{A}(A)+F_{P}(P)-F_{d}(d,D)-F_{D}(D)}{F_{h}(h,D)}
$$
$$
h \epsilon ℤ | h=[1...255]
$$
$$
a \epsilon ℤ | a=[5...190]
$$
$$
d \epsilon ℤ | d=[5...250]
$$
$$
A \epsilon ℤ | A=[1...100]
$$
$$
D \epsilon ℤ | D=[1...100]
$$
$$
P \epsilon ℤ | P=5*m, m \epsilon ℤ | m=[2...50]
$$

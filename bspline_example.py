import numpy as np

# cv = np.array of 3d control vertices
# n = number of samples (default: 100)
# d = curve degree (default: cubic)
# closed = is the curve closed (periodic) or open? (default: open)
def bspline(cv, n=100, d=3, closed=False):

    # Create a range of u values
    count = len(cv)
    knots = None
    u = None
    if not closed:
        u = np.arange(0,n,dtype='float')/(n-1) * (count-d)
        rr = list(range(count-d+1))
        # knots = np.array([0]*d + rr + [count-d]*d,dtype='int')
        knots = np.array(list(range(len(cv)+d+1))) - d
        print(knots)
    else:
        u = ((np.arange(0,n,dtype='float')/(n-1) * count) - (0.5 * (d-1))) % count # keep u=0 relative to 1st cv
        knots = np.arange(0-d,count+d+d-1,dtype='int')
        print(knots)


    # Simple Cox - DeBoor recursion
    def coxDeBoor(u, k, d):

        # Test for end conditions
        if (d == 0):
            if (knots[k] <= u and u < knots[k+1]):
                return 1
            return 0

        Den1 = knots[k+d] - knots[k]
        Den2 = knots[k+d+1] - knots[k+1]
        Eq1  = 0;
        Eq2  = 0;

        if Den1 > 0:
            Eq1 = ((u-knots[k]) / Den1) * coxDeBoor(u,k,(d-1))
        if Den2 > 0:
            Eq2 = ((knots[k+d+1]-u) / Den2) * coxDeBoor(u,(k+1),(d-1))

        return Eq1 + Eq2


    # Sample the curve at each u value
    samples = np.zeros((n,3))
    for i in range(n):
        if not closed:
            if u[i] == count-d:
                samples[i] = np.array(cv[-1])
            else:
                for k in range(count):
                    samples[i] += coxDeBoor(u[i],k,d) * cv[k]

        else:
            for k in range(count+d):
                samples[i] += coxDeBoor(u[i],k,d) * cv[k%count]


    return samples




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def test(closed):
        cv = np.array([[ 50.,  25.,  -0.],
               [ 59.,  12.,  -0.],
               [ 50.,  10.,   0.],
               [ 57.,   2.,   0.],
               [ 40.,   4.,   0.],
               [ 40.,   14.,  -0.]])

        p = bspline(cv,closed=closed)
        x,y,z = p.T
        cv = cv.T
        plt.plot(cv[0],cv[1], 'o-', label='Control Points')
        plt.plot(x,y,'k-',label='Curve')
        plt.minorticks_on()
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(35, 70)
        plt.ylim(0, 30)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    test(False)


import numpy as np

def b_spline_basis(i, k, t, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i+1] else 0.0
    
    first_term = 0.0
    second_term = 0.0
    
    if knots[i+k] - knots[i] != 0:
        first_term = (t - knots[i]) / (knots[i+k] - knots[i]) * b_spline_basis(i, k-1, t, knots)
        
    if knots[i+k+1] - knots[i+1] != 0:
        second_term = (knots[i+k+1] - t) / (knots[i+k+1] - knots[i+1]) * b_spline_basis(i+1, k-1, t, knots)
    
    return first_term + second_term

def b_spline_curve(t, control_points, degree, knots):
    n = len(control_points) - 1
    curve_point = np.zeros_like(control_points[0], dtype=float)
    for i in range(n+1):
        basis = b_spline_basis(i, degree, t, knots)
        curve_point += basis * control_points[i]
    return curve_point

# Example usage:
control_points = np.array([[1, 1], [1, 3], [3, 5], [5, 0], [7, 2], [9, 6]], dtype=float)
degree = 3
knots = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)

# Evaluate the B-spline curve at parameter t (t should be in the range [degree, n+1])
# Range of t values
t_values = np.linspace(degree, len(control_points), 1000)

# Calculate points on the B-spline curve for each t value
curve_points = np.array([b_spline_curve(t, control_points, degree, knots) for t in t_values])

# Plot the B-spline curve
plt.figure(figsize=(8, 6))
plt.plot(curve_points[:, 0], curve_points[:, 1], label='B-spline Curve', color='b')
plt.scatter(control_points[:, 0], control_points[:, 1], c='r', label='Control Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('B-spline Curve')
plt.legend()
plt.grid(True)
plt.show()





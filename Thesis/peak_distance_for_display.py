d85 = np.max(dose_map)*0.85
isodose = ax.contour(y, x, dose_map, levels = [d85], colors  = "blue")
lines = []
#getting coordinates from isodose lines
for line in isodose.collections[0].get_paths():
    if line.vertices.shape[0] > 500: #less than hundred points is not a dose peak edge
            lines.append(line.vertices)
odd_i = 1
#looping over the center of each quadrat 
for i in range(kernel_size//2, pooled_dose.shape[0]*kernel_size , kernel_size):
    odd_j = 1
    for j in range(kernel_size//2,pooled_dose.shape[1]*kernel_size, kernel_size):  
        min_d = 1e6  #Large distance not possible to surpass
        #centre of the kernel 
        centre = [i + kernel_size/2-kernel_size//2, j + kernel_size/2-kernel_size//2] # [x-axis, y-axis]
        for line in lines: #getting information
            x = line[:,1] #as vertices is (column, row) we need to get index 1
            y = line[:,0]
            d = np.sqrt((x -centre[0])**2 + (y-centre[1])**2)
            tmp = np.min(d)
            if tmp < min_d:
                min_d = tmp
                idx_tmp = np.argwhere(d == min_d)
        #if the quadrat is located within a peak, then the distance is 0
        if dose_map[i,j] > d85: #assumes only 5 Gy irradiated films
            if kernel_size%2 == 0:
                dist[i//kernel_size, j//kernel_size] = 0
            else:
                dist[i - kernel_size//2*odd_i,j - kernel_size//2*odd_j] = 0
        else:
            if kernel_size % 2 == 0:
                dist[i//kernel_size, j//kernel_size] = min_d
            else:
                dist[i - kernel_size//2*odd_i,j - kernel_size//2*odd_j] = \
                 min_d
        odd_j += 2
    odd_i += 2
"""
PCA Projection and Visualization Pipeline for MD Trajectories

Description:
    This script performs Principal Component Analysis (PCA) on molecular dynamics (MD) trajectories 
    to analyze and visualize large-scale conformational changes. It projects both a reference 
    trajectory and one or more projection trajectories into the same PCA space and generates 2D 
    and 3D scatter plots to compare their conformational landscapes over time.

Modules and Workflow:
    - `proj_pca()`: 
        • Loads the reference and projection trajectories.
        • Performs PCA on the reference trajectory.
        • Projects all trajectories into the PCA space (first 3 and 6 components).
        • Saves the PCA-reduced coordinates to `.csv` files for further plotting.

    - `plot2dfunc()` and `plot3dfunc()`:
        • Create 2D and 3D scatter plots of PCA projections with time as the color dimension.
        • Overlay grid lines and calculate the number of unique occupied grid squares (2D only).
        • Save high-resolution images of each plot.

    - `proj_plot()`:
        • Combines reduced data, performs labeling and formatting.
        • Calls the plotting functions for selected PCA component combinations.
        • Merges projected PCA data across all simulations into one combined `.csv`.

Requirements:
    - Python packages: `mdtraj`, `matplotlib`, `pandas`, `scikit-learn`, `numpy`
    - Input files:
        • `path_dcd_ref`: Reference trajectory (.dcd)
        • `path_pdb`: Topology for reference (.pdb)
        • `path_dcd_proj`: List of projection trajectories (.dcd)
        • `pdb_proj`: Topology for projection (.pdb)
    - Ensure atomic selections used in `select` and `select_proj` return the same number/type of atoms.

User Parameters:
    - `name`: Label prefix for all saved output files.
    - `savedir`: Directory path for saving outputs.
    - `stridenum`, `stridenum_proj`: Frame skipping factor for loading trajectories.
    - `format`: Image file format (e.g., "png", "svg").
    - `font_size`: Font size for plot labels.
    - `screetitle`, `pctitle`: Plot titles.
    - `select`, `select_proj`: MDTraj-style atom selections for PCA.
    - `microsec`: Maximum simulation time (in μs) used for color bar scaling.
    - `simulation_timestep`: Time between MD frames (in femtoseconds).
    - `dcd_save_freq`: Frequency (in steps) at which trajectory frames were saved.
    - `show_plots`: Whether to display the plots after saving.

Outputs:
    - CSV files:
        • `*-proj-reduced_cartesian.csv`: First 3 PCA coordinates per projection.
        • `*-proj-reduced_cartesian_big.csv`: First 6 PCA coordinates.
        • `*-all-proj-combine-reduced_cartesian.csv`: Combined PCA projection data.
        • `*-variance_ratio.csv`: Variance explained by each PC (if generated beforehand).
    - Image files:
        • `*-proj-2D-PCA-12-grid.png`: 2D PC1 vs PC2 projection.
        • `*-proj-2D-PCA-13-grid.png`: 2D PC1 vs PC3 projection.
        • `*-proj-2D-PCA-23-grid.png`: 2D PC2 vs PC3 projection.
        • `*-proj-3D-PCA.png`: 3D projection colored by time.

Notes:
    - The script assumes that PCA has not been previously computed; it recomputes it each time.
    - Use `select` and `select_proj` carefully to maintain consistent atom mappings across systems.
    - The function `truncate_colormap` is used to create visually distinct colormaps with cutoff ranges.
"""



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot2dfunc(x, y, xx, yy, n1, n2, savedir, name, format, font_size,
               pctitle, microsec, time, replica, show_plots,
               simulation_timestep, dcd_save_freq, grid_size=0.1):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.pyplot import cm
    import math

    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.axes()

    # Load variance explained
    path = savedir + name + '-variance_ratio.csv'
    df = pd.read_csv(path, index_col=0)
    df.columns = ["variance"]
    vars = df["variance"] * 100

    # Time scaling
    fgds = dcd_save_freq * simulation_timestep * 1e-9
    time = (time + 1) * fgds

    # Main dataset scatter
    cmap = plt.get_cmap('spring')
    new_cmap = truncate_colormap(cmap, 0.2, 0.85)
    p = ax1.scatter(x, y, marker='x', c=time, cmap=new_cmap)

    # Axis labels
    string1 = f"PC{n1} ({round(vars[int(n1)-1], 1)}%)"
    string2 = f"PC{n2} ({round(vars[int(n2)-1], 1)}%)"
    ax1.set(xlabel=string1, ylabel=string2)
    ax1.set_title(pctitle, wrap=True)

    # Colorbar
    microsec = max(time)
    tick = math.ceil(microsec / 4) / 2
    if tick > microsec:
        tick = microsec / 4
    plt.colorbar(p, pad=0.1, shrink=0.5, ticks=np.arange(0, microsec*1.001, tick), label=r'$\mu$s')


    # Overlay projections
    color = iter(cm.rainbow(np.linspace(0, 1, len(xx))))
    cmap_list = ['summer', 'winter']
    for i in range(len(xx)):
        c = next(color)
        time_proj = np.arange(len(xx[i]))  # or use actual time if available
        cmap = plt.get_cmap(cmap_list[i % 2])
        new_cmap = truncate_colormap(cmap, 0.0, 0.7)
        ax1.scatter(xx[i], yy[i], marker='x', c=time_proj, cmap=new_cmap)

    # Determine plotting limits
    all_x = np.concatenate([x] + xx)
    all_y = np.concatenate([y] + yy)
    x_min, x_max = np.floor(all_x.min()), np.ceil(all_x.max())
    y_min, y_max = np.floor(all_y.min()), np.ceil(all_y.max())
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    # Grid overlay
    for gx in np.arange(x_min, x_max + grid_size, grid_size):
        ax1.axvline(gx, color='gray', linewidth=0.3, linestyle='--', zorder=0)
    for gy in np.arange(y_min, y_max + grid_size, grid_size):
        ax1.axhline(gy, color='gray', linewidth=0.3, linestyle='--', zorder=0)

    # Count occupied grid cells
    def count_bins(xdata, ydata):
        bins_x = ((np.array(xdata) - x_min) / grid_size).astype(int)
        bins_y = ((np.array(ydata) - y_min) / grid_size).astype(int)
        coords = set(zip(bins_x, bins_y))
        return len(coords)

    base_count = count_bins(x, y)
    print(f"[GRID] Base trajectory occupies {base_count} grid squares.")

    for i in range(len(xx)):
        proj_count = count_bins(xx[i], yy[i])
        print(f"[GRID] Projection {i} occupies {proj_count} grid squares.")

    # Final plot adjustments
    plt.tight_layout()
    ax1.set_box_aspect(1)
    savepath = savedir + name + f'-proj-2D-PCA-{n1}{n2}-grid.{format}'
    plt.savefig(savepath, format=format, dpi=1000)

    if show_plots:
        plt.show()

    plt.close('all')
    return

def plot3dfunc(x,y,z,xx,yy,zz,savedir,name,format,font_size,pctitle,microsec,time,replica,show_plots,simulation_timestep,dcd_save_freq):
	"2d plot function"
	import mdtraj as md
	from sklearn.decomposition import PCA
	from sklearn.cluster import KMeans
	from sklearn.preprocessing import StandardScaler
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	from sklearn.metrics import pairwise_distances_argmin_min
	from matplotlib.ticker import FuncFormatter
	import pandas as pd
	import matplotlib.pyplot as plt
	import numpy as np
	from itertools import combinations
	import math
	from matplotlib.pyplot import cm
	plt.rcParams.update({'font.size': font_size})
	fgds = dcd_save_freq*simulation_timestep*(10**(-9))
	fig=plt.figure()
	ax1 = fig.add_subplot(projection = '3d')
	p=ax1.scatter(x,y,z, marker='x', c=(time +1)*fgds, cmap = 'winter')
	title = pctitle
	ax1.set_title(title, wrap=True)

	color = iter(cm.rainbow(np.linspace(0, 1, len(xx))))
	for i in range(len(xx)):
		c = next(color)
		ax1.scatter(xx[i],yy[i],zz[i],marker='o', c=c)
		ax1.grid(False)

	#colorbar
	tick = math.ceil(microsec/4)/2
	if tick > microsec:
		tick = microsec/4

	fig.tight_layout()
	plt.colorbar(p, pad=0.1, shrink=0.5, ticks=np.arange(0, microsec*1.001, tick), label=r'$\mu$s')
	#plt.colorbar(q, ticks=np.arange(0,microsec*1.001,tick) )
	#plt.colorbar(q, ticks=np.arange(1,len(replica),len(replica)/7), label='Reaction Coordinate' )
	plt.tight_layout()
	savepath = savedir + name + '-proj-3D-PCA.'+format
	plt.savefig(savepath, format=format, dpi=1000)
	if show_plots == True:
		plt.show()
	plt.close('all')
	return
	return


def proj_pca(path_dcd_ref,path_dcd_proj,path_pdb,pdb_proj,name,savedir,stridenum,stridenum_proj,format,font_size,screetitle,select,select_proj,show_plots,microsec,simulation_timestep,dcd_save_freq,pctitle):
	"Makes scree plot"
	import mdtraj as md
	from sklearn.decomposition import PCA
	from sklearn.cluster import KMeans
	from sklearn.preprocessing import StandardScaler
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	from sklearn.metrics import pairwise_distances_argmin_min
	from matplotlib.ticker import FuncFormatter
	import pandas as pd
	import matplotlib.pyplot as plt
	import numpy as np
	from itertools import combinations
	import math

	traj=md.load_dcd(path_dcd_ref, top=path_pdb,stride=stridenum)
	traj2 = []
	for path in path_dcd_proj:
		traj2.append(md.load_dcd(path,top=pdb_proj,stride=stridenum_proj))

	query = traj.topology.select(select)
	query2 = []
	for trajj in traj2:
		query2.append(trajj.topology.select(select_proj))

	traj=traj.atom_slice(query)
	traj.superpose(traj,0)

	for i in range(len(traj2)):
		traj2[i] = traj2[i].atom_slice(query2[i])
		traj2[i].superpose(traj,0)

	pca1=PCA(n_components=3)
	fgds = pca1.fit(traj.xyz.reshape(traj.n_frames, traj.n_atoms*3))
	reduced_cartesian_proj = []
	for trajj in traj2:
		reduced_cartesian_proj.append(fgds.transform(trajj.xyz.reshape(trajj.n_frames, trajj.n_atoms*3)))

	for i in range(len(traj2)):
		df=pd.DataFrame(reduced_cartesian_proj[i])
		savepath = savedir + name + '-' + str(i) + '-proj-reduced_cartesian.csv'
		df.to_csv(savepath)

	pca2=PCA(n_components=6)
	fgds = pca1.fit(traj.xyz.reshape(traj.n_frames, traj.n_atoms*3))
	reduced_cartesian2_proj = []
	for trajj in traj2:
		reduced_cartesian2_proj.append(fgds.transform(trajj.xyz.reshape(trajj.n_frames, trajj.n_atoms*3)))

	for i in range(len(traj2)):
		df=pd.DataFrame(reduced_cartesian_proj[i])
		savepath = savedir + name + '-' + str(i) + '-proj-reduced_cartesian_big.csv'
		df.to_csv(savepath)
	return

def proj_plot(path_dcd_ref,path_dcd_proj,path_pdb,pdb_proj,name,savedir,stridenum,format,font_size,screetitle,select,show_plots,microsec,simulation_timestep,dcd_save_freq,pctitle):
	#plots
	import mdtraj as md
	from sklearn.decomposition import PCA
	from sklearn.cluster import KMeans
	from sklearn.preprocessing import StandardScaler
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	from sklearn.metrics import pairwise_distances_argmin_min
	from matplotlib.ticker import FuncFormatter
	import pandas as pd
	import matplotlib.pyplot as plt
	import numpy as np
	from itertools import combinations
	import math
	thedfs = []

	path = 'JM_op_8on0_ter_cov2-reduced_cartesian.csv'
	df = pd.read_csv(path)
	headers= ["time","pc1","pc2","pc3"]
	df.columns = headers

	thedfs.append(df)

	projdfs = []
	for i in range(len(path_dcd_proj)):
		path = savedir + name + '-' + str(i) + '-proj-reduced_cartesian_big.csv'
		#path = savedir + name + '-0-proj-reduced_cartesian_big.csv'
		df2 = pd.read_csv(path)
		headers= ["time","pc1","pc2","pc3"]
		df2.columns = headers
		thedfs.append(df2)
		projdfs.append(df2)

	concat = pd.concat(thedfs)

	savepath = savedir + name + '-all-proj-combine-reduced_cartesian.csv'
	concat.to_csv(savepath)

	time = df['time']

	replica = []
	for dff in projdfs:
		replica.append(dff['time'])

	x = df['pc1']
	y = df['pc2']
	z = df['pc3']

	xx = []
	yy = []
	zz = []
	for dff in projdfs:
		xx.append(dff['pc1'])
		yy.append(dff['pc2'])
		zz.append(dff['pc3'])

	plot2dfunc(x,y,xx,yy,'1','2',savedir,name,format,font_size,pctitle,microsec,time,replica,show_plots,simulation_timestep,dcd_save_freq)
	plot2dfunc(x,z,xx,zz,'1','3',savedir,name,format,font_size,pctitle,microsec,time,replica,show_plots,simulation_timestep,dcd_save_freq)
	plot2dfunc(y,z,yy,zz,'2','3',savedir,name,format,font_size,pctitle,microsec,time,replica,show_plots,simulation_timestep,dcd_save_freq)
	plot3dfunc(x,y,z,xx,yy,zz,savedir,name,format,font_size,pctitle,microsec,time,replica,show_plots,simulation_timestep,dcd_save_freq)
	return


if __name__ == "__main__":
    # Example: Adjust these paths and parameters to your actual setup
    path_dcd_ref = "cov2_0_unwrap.dcd"
    path_dcd_proj = ["cov2_8_unwrap.dcd"]
    path_pdb = "cov2_kc_0.pdb"
    pdb_proj = "cov2_kc_8.pdb"
    name = "8on0"
    savedir = "/Users/tasnims/Desktop/JM_op_grid/"
    stridenum = 1
    format = "png"
    font_size = 14
    screetitle = "Scree plot"
    select = "name CA"
    show_plots = False
    microsec = 0.2  # or whatever makes sense
    simulation_timestep = 2  # fs
    dcd_save_freq = 10000
    pctitle = "2D PCA Projection"

    proj_plot(
        path_dcd_ref,
        path_dcd_proj,
        path_pdb,
        pdb_proj,
        name,
        savedir,
        stridenum,
        format,
        font_size,
        screetitle,
        select,
        show_plots,
        microsec,
        simulation_timestep,
        dcd_save_freq,
        pctitle
    )


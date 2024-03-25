        if plot_intensity_radii:

            fig, axes = plt.subplots()

            axes.plot(r, I_int, marker = 'o')

            print("plotting intensity")
            axes.set_xscale("log")

            plt.show()
            plt.close()
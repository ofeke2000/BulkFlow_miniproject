imports

setting up timers and loggers

data loading

calculate overdensity

for halo in closest n densities to 0 loop:{
    create 2 masks
    for radius in range (min radius, max radius, radius jumps){  #perhaps this loop can be a function
        calculate bulk flow around halo for the radius for both masks
    }
}

calculate average bulk flow for each radius

plot it

save data and plot
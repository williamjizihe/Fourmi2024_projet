import numpy as np
import maze
import pheromone
from ants import Colony
import pygame as pg
import sys
import time

from mpi4py import MPI

globCom     = MPI.COMM_WORLD
globRank    = globCom.rank
globNbp     = globCom.size

subCom      = globCom.Split(color=0 if globRank == 0 else 1)
subRank     = subCom.rank
subNbp      = subCom.size

UNLOADED, LOADED = False, True

exploration_coefs = 0.

size_laby = 25, 25
if len(sys.argv) > 2:
    size_laby = int(sys.argv[1]),int(sys.argv[2])

resolution = size_laby[1]*8, size_laby[0]*8
nb_ants = size_laby[0]*size_laby[1]//4 // subNbp * subNbp
max_life = 500

if len(sys.argv) > 3:
    max_life = int(sys.argv[3])
pos_food = size_laby[0]-1, size_laby[1]-1
pos_nest = 0, 0
unloaded_ants = np.array(range(nb_ants))
alpha = 0.9
beta  = 0.99
if len(sys.argv) > 4:
    alpha = float(sys.argv[4])
if len(sys.argv) > 5:
    beta = float(sys.argv[5])

food_counter = 0
if globRank == 0: # Display
    pg.init()
    screen = pg.display.set_mode(resolution)
    a_maze = maze.Maze(size_laby, 12345, display=True)
    mazeImg = a_maze.display()
    fps_file = open("fps.txt", "w")
    pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
    ants = Colony(nb_ants, pos_nest, max_life, display=True)
    snapshop_taken = False
    f_c = np.empty(1, dtype=np.int)
    
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit(0)

        deb = time.time()
        globCom.Recv([pherom.pheromon, MPI.DOUBLE], source=1)
        globCom.Recv([ants.historic_path, MPI.INT16_T], source=1)
        globCom.Recv([ants.directions, MPI.INT8_T], source=1)
        globCom.Recv([ants.age, MPI.INT64_T], source=1)
        globCom.Recv([f_c, MPI.INT], source=1)
        
        pherom.display(screen)
        screen.blit(mazeImg, (0, 0))
        ants.display(screen)
        pg.display.update()
        
        end = time.time()
        food_counter = f_c[0]
        if food_counter == 1 and not snapshop_taken:
            pg.image.save(screen, "MyFirstFood.png")
            snapshop_taken = True
        if food_counter >= 2000:
            break
        fps_file.write(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter:7d}\n")
        # pg.time.wait(500)
        # print(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter:7d}", end='\r')
else: 
    # All ranks different from 0 do the calculation
    a_maze = maze.Maze(size_laby, 12345, display=False)
    pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
    ants = Colony(nb_ants // subNbp, pos_nest, max_life, display=False)
    
    glob_historic_path = np.empty((nb_ants, max_life+1, 2), dtype=np.int16) if subRank == 0 else None
    glob_directions = np.empty(nb_ants, dtype=np.int8) if subRank == 0 else None
    glob_age = np.empty(nb_ants, dtype=np.int64) if subRank == 0 else None
    glob_food_counter = np.empty(1, dtype=np.int)
    
    while True:
        food_counter, variation_pheromon, variation_pos = \
            ants.advance(a_maze, pos_food, pos_nest, pherom, food_counter)
        sum_variation_pheromon = np.zeros_like(variation_pheromon)
        sum_variation_pos = np.zeros_like(variation_pos)
        # Syncronize the pheromones
        subCom.Allreduce([variation_pheromon, MPI.DOUBLE], [sum_variation_pheromon, MPI.DOUBLE], op=MPI.SUM)
        subCom.Allreduce([variation_pos, MPI.INT], [sum_variation_pos, MPI.INT], op=MPI.SUM)
        pherom.pheromon += sum_variation_pheromon /  np.maximum(sum_variation_pos, 1)
        
        pherom.do_evaporation(pos_food)
        
        # Gather the results
        subCom.Gather([ants.historic_path, MPI.INT16_T], [glob_historic_path, MPI.INT16_T], root=0)
        subCom.Gather([ants.directions, MPI.INT8_T], [glob_directions, MPI.INT8_T], root=0)
        subCom.Gather([ants.age, MPI.INT64_T], [glob_age, MPI.INT64_T], root=0)
        subCom.Allreduce([np.array([food_counter], dtype=np.int), MPI.INT], [glob_food_counter, MPI.INT], op=MPI.SUM)
        if subRank == 0:
            req1 = globCom.Isend([pherom.pheromon, MPI.DOUBLE], dest=0)
            req2 = globCom.Isend([glob_historic_path, MPI.INT16_T], dest=0)
            req3 = globCom.Isend([glob_directions, MPI.INT8_T], dest=0)
            req4 = globCom.Isend([glob_age, MPI.INT64_T], dest=0)
            req5 = globCom.Isend([glob_food_counter, MPI.INT], dest=0)
            MPI.Request.Waitall([req1, req2, req3, req4, req5])
        if glob_food_counter[0] >= 2000:
            break

globCom.Barrier()
if globRank == 0:
    fps_file.close()
    pg.image.save(screen, "Final.png")
    pg.quit()
    
subCom.Free()
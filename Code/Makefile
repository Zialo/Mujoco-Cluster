# MAKE DATASETS
push_ball:
	python3 dataset_generator_push_ball.py 500 100 0.8 data_4	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	
stick_puss:
	python3 dataset_generator_stick_puss.py 10 150 0.8 data	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]

button_press:
	python3 dataset_generator_button_press.py 150 100 0.8 Button1	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	python3 dataset_generator_button_press.py 150 100 0.8 Button2	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	python3 dataset_generator_button_press.py 150 100 0.8 Button3	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	python3 dataset_generator_button_press.py 150 100 0.8 Button4	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	python3 dataset_generator_apilator.py
	
window_open:	
	python3 dataset_generator_window_open.py 250 150 0.8 window1
	python3 dataset_generator_window_open.py 250 150 0.8 window2
	python3 dataset_generator_window_open.py 250 150 0.8 window3
	python3 dataset_generator_window_open.py 250 150 0.8 window4
	#python3 dataset_generator_apilator.py
	
basketball:	
	python3 dataset_generator_basketball.py 150 5 0.8 basketball1
	python3 dataset_generator_basketball.py 150 250 0.8 basketball2
	python3 dataset_generator_basketball.py 150 250 0.8 basketball3
	python3 dataset_generator_basketball.py 150 250 0.8 basketball4
	python3 dataset_generator_apilator.py

basketball_depth:	
	python3 -m memory_profiler dataset_generator_basketball_depth.py 2 3 0.5 basketball1
	
	

shelf_place:
	python3 dataset_generator_shelf_place.py 5 200 0.8 SPlace_5	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	python3 dataset_generator_shelf_place.py 25 200 0.8 SPlace_25	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	python3 dataset_generator_shelf_place.py 50 200 0.8 SPlace_50_1	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	python3 dataset_generator_shelf_place.py 50 200 0.8 SPlace_50_2	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	python3 dataset_generator_apilator.py
	python3 dataset_generator_shelf_place.py 50 200 0.8 SPlace_50_3
	python3 dataset_generator_apilator_2.py
	python3 dataset_generator_shelf_place.py 50 200 0.8 SPlace_50_4	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	python3 dataset_generator_apilator_3.py
	
# MAKE SIMULATIONS

# PUSH BALL
simulacion_t-5:
	python3 Simulation_t-5.py 0 50 100 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_basic_v2:
	python3 Simulation_basic_v2.py 0 100 150 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_basic_v2_ng:
	python3 Simulation_basic_v2_noGripper.py 0 100 150 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]

# BUTTON PRESS	
simulacion_t-5_button:
	python3 Simulation_basic_v2_button.py 0 50 100 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_t-5_button_ng:
	python3 Simulation_basic_v2_button_noGripper.py 0 100 100 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_t-1_button_ng:
	python3 Simulation_basic_v2_button_noGripper_1.py 0 100 100 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_t-1_button_imgSep:
	python3 Simulation_basic_v3_button_imgSep.py 0 100 150 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]

# WINDOW OPEN
simulacion_t-5_window_ng:
	python3 Simulation_basic_v2_window_noGripper.py 0 100 100 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_t-1_window_ng:
	python3 Simulation_basic_v2_window_noGripper_1.py 0 100 200 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_t-1_window_obs:
	python3 Simulation_basic_v3_window_obs.py 0 100 200 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_t-1_window_imgSep:
	python3 Simulation_basic_v3_window_imgSep.py 0 100 200 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_t-1_window_imgSep_obs2:
	python3 Simulation_basic_v4_window_obs2_imgSep.py 0 2 3 1 True window_2-obs_042  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]

simulacion_t-1_window_imgSep_obs:
	python3 Simulation_basic_v3_window_obs_imgSep.py 0 100 200 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_normalizada_entradas:
	python3 Simulation_basic_v6_window_obs2_imgSep_norm.py 0 100 200 1 True Window_Normalized_Obs-2V3
	
simulacion_normalizada_entradas2:
	python3 Simulation_basic_v7_window_obs2_imgSep_norm.py 0 100 200 1 True Window_Normalized_Obs-2V3

simulacion_normalizada_todo:
	python3 Simulation_basic_v4_window_obs2_imgSep_normalized.py 0 100 200 1 True Window_Normalized_Obs-2V2

# BASKETBALL
simulacion_t-1_basketball_imgSep:
	python3 Simulation_basic_basketball_imgSep.py 0 100 300 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]	
	
# TFM
TFM_BALL_INDIVIDUAL:
	python3 Simulation_TFM_Ball_Individual.py 0 20 100 1 True 'MetaWorld_0258_Top.zip'
	python3 Simulation_TFM_Ball_Individual.py 0 20 100 1 True 'MetaWorld_0204_Corner.zip'
	python3 Simulation_TFM_Ball_Individual.py 0 20 100 1 True 'MetaWorld_0254_Corner2.zip'
	python3 Simulation_TFM_Ball_Individual.py 0 20 100 1 True 'MetaWorld_0291_Corner3.zip'
	python3 Simulation_TFM_Ball_Individual.py 0 20 100 1 True 'MetaWorld_0591_Gripper.zip'
	python3 Simulation_TFM_Ball_Individual.py 0 20 100 1 True 'MetaWorld_0214_BehindGripper.zip'
TFM_BALL_DOBLE:
	python3 Simulation_TFM_Ball_Doble.py 0 20 100 1 True 'MetaWorld_0158_Corner+BehindGripper.zip'
	python3 Simulation_TFM_Ball_Doble.py 0 20 100 1 True 'MetaWorld_0193_Top+Corner2.zip'
	python3 Simulation_TFM_Ball_Doble.py 0 20 100 1 True 'MetaWorld_0197_Top+Corner.zip'
	python3 Simulation_TFM_Ball_Doble.py 0 20 100 1 True 'MetaWorld_0214_Corner+Corner3.zip'
	python3 Simulation_TFM_Ball_Doble.py 0 20 100 1 True 'MetaWorld_0218_Corner+Corner2.zip'
	python3 Simulation_TFM_Ball_Doble.py 0 20 100 1 True 'MetaWorld_0228_Corner2+Corner3.zip'
	python3 Simulation_TFM_Ball_Doble.py 0 20 100 1 True 'MetaWorld_0381_Top+Corner3.zip'
TFM_BALL_RESTANTES:
	python3 Simulation_TFM_Ball_Triple.py 0 20 100 1 True 'MetaWorld_0162_Top+Corner2+BehindGripper.zip'
	python3 Simulation_TFM_Ball_Triple.py 0 20 100 1 True 'MetaWorld_02_Top+Corner+Corner2.zip'
	python3 Simulation_TFM_Ball_Triple.py 0 20 100 1 True 'MetaWorld_015_Top+Corner+BehindGripper.zip'
	python3 Simulation_TFM_Ball_Cuadruple.py 0 20 100 1 True 'MetaWorld_0177_Top+Corner+Corner2+BehindGripper.zip'
	python3 Simulation_TFM_Ball_Quintuple.py 0 20 100 1 True 'MetaWorld_0143_Top+Corner+Corner2+Corner3+BehindGripper.zip'
	python3 Simulation_TFM_Ball.py 0 20 100 1 True 'MetaWorld_0166_ALL.zip'
	
TFM_BUTTON_INDIVIDUAL:
	python3 Simulation_TFM_Button_Individual.py 0 20 100 1 True 'MetaWorld_04_Top.zip'
	python3 Simulation_TFM_Button_Individual.py 0 20 100 1 True 'MetaWorld_26.299_Corner.zip'
	python3 Simulation_TFM_Button_Individual.py 0 20 100 1 True 'MetaWorld_0452_Corner2.zip'
	python3 Simulation_TFM_Button_Individual.py 0 20 100 1 True 'MetaWorld_0451_Corner3.zip'
	python3 Simulation_TFM_Button_Individual.py 0 20 100 1 True 'MetaWorld_0496_BehindGripper.zip'
	python3 Simulation_TFM_Button_Individual.py 0 20 100 1 True 'MetaWorld_0713_Gripper.zip'
TFM_BUTTON_DOBLE:
	python3 Simulation_TFM_Button_Doble.py 0 20 100 1 True 'MetaWorld_0418_Top+Corner3.zip'
	python3 Simulation_TFM_Button_Doble.py 0 20 100 1 True 'MetaWorld_0379_Top+Corner.zip'
	python3 Simulation_TFM_Button_Doble.py 0 20 100 1 True 'MetaWorld_0374_Top+Corner2.zip'
	python3 Simulation_TFM_Button_Doble.py 0 20 100 1 True 'MetaWorld_0354_Corner+Corner3.zip'
	python3 Simulation_TFM_Button_Doble.py 0 20 100 1 True 'MetaWorld_0349_Corner+Corner2.zip'
	python3 Simulation_TFM_Button_Doble.py 0 20 100 1 True 'MetaWorld_0403_Corner2+Corner3.zip'
	python3 Simulation_TFM_Button_Doble.py 0 20 100 1 True 'MetaWorld_0311_Corner+BehindGripper.zip'
TFM_BUTTON_RESTANTES:
	python3 Simulation_TFM_Button_Triple.py 0 20 100 1 True 'MetaWorld_0369_Top+Corner+Corner2.zip'
	python3 Simulation_TFM_Button_Triple.py 0 20 100 1 True 'MetaWorld_034_Top+Corner+BehindGripper'
	python3 Simulation_TFM_Button_Triple.py 0 20 100 1 True 'MetaWorld_034_Top+Corner2+BehindGripper.zip'
	python3 Simulation_TFM_Button_Cuadruple.py 0 20 100 1 True 'MetaWorld_0295_Top+Corner+Corner2+BehindGripper.zip'
	python3 Simulation_TFM_Button_Quintuple.py 0 20 100 1 True 'MetaWorld_0298_Top+Corner+Corner2+Corner3+BehindGripper.zip'
	python3 Simulation_TFM_Button.py 0 20 100 1 True 'MetaWorld_0324_ALL.zip'	
TFM_BUTTON_EXTRAS:
	python3 Simulation_TFM_Button_Triple.py 0 20 100 1 True 'MetaWorld_034_Top+Corner2+BehindGripper.zip'
	python3 Simulation_TFM_Button_Cuadruple.py 0 20 100 1 True 'MetaWorld_0295_Top+Corner+Corner2+BehindGripper.zip'
	python3 Simulation_TFM_Button_Quintuple.py 0 20 100 1 True 'MetaWorld_0298_Top+Corner+Corner2+Corner3+BehindGripper.zip'
	python3 Simulation_TFM_Button.py 0 20 100 1 True 'MetaWorld_0324_ALL.zip'
	
TFM_WINDOW_INDIVIDUAL:
	python3 Simulation_TFM_Window_Individual.py 0 20 200 1 True 'MetaWorld_0463_Corner.zip'
	python3 Simulation_TFM_Window_Individual.py 0 20 200 1 True 'MetaWorld_0509_Top.zip'
	python3 Simulation_TFM_Window_Individual.py 0 20 200 1 True 'MetaWorld_0538_BehindGripper.zip'
	python3 Simulation_TFM_Window_Individual.py 0 20 200 1 True 'MetaWorld_0519_Corner3.zip'
	python3 Simulation_TFM_Window_Individual.py 0 20 200 1 True 'MetaWorld_0792_Gripper.zip'
	python3 Simulation_TFM_Window_Individual.py 0 20 200 1 True 'MetaWorld_0616_Corner2.zip'
TFM_WINDOW_DOBLE:
	python3 Simulation_TFM_Window_Individual.py 0 20 200 1 True 'MetaWorld_0616_Corner2.zip'
	python3 Simulation_TFM_Window_Individual.py 0 20 200 1 True 'MetaWorld_0792_Gripper.zip'
	python3 Simulation_TFM_Window_Doble.py 0 20 200 1 True 'MetaWorld_0938_Corner+Corner3.zip'
	python3 Simulation_TFM_Window_Doble.py 0 20 200 1 True 'MetaWorld_0662_Top+Corner3.zip'
	python3 Simulation_TFM_Window_Doble.py 0 20 200 1 True 'MetaWorld_0654_Corner+Corner2.zip'
	python3 Simulation_TFM_Window_Doble.py 0 20 200 1 True 'MetaWorld_0643_Top+Corner2.zip'
	python3 Simulation_TFM_Window_Doble.py 0 20 200 1 True 'MetaWorld_0596_Top+Corner.zip'
	python3 Simulation_TFM_Window_Doble.py 0 20 200 1 True 'MetaWorld_0535_Corner2+Corner3.zip'
	python3 Simulation_TFM_Window_Doble.py 0 20 200 1 True 'MetaWorld_043_Corner+BehindGripper.zip'
TFM_WINDOW_RESTANTES:
	python3 Simulation_TFM_Window_Triple.py 0 20 200 1 True 'MetaWorld_0488_Top+Corner+Corner2.zip'
	python3 Simulation_TFM_Window_Triple.py 0 20 200 1 True 'MetaWorld_0332_Top+Corner+BehindGripper.zip'
	python3 Simulation_TFM_Window_Triple.py 0 20 200 1 True 'MetaWorld_0329_Top+Corner2+BehindGripper.zip'
	python3 Simulation_TFM_Window_Cuadruple.py 0 20 100 1 True 'MetaWorld_054_Top+Corner+Corner2+BehindGripper.zip'
	python3 Simulation_TFM_Window_Quintuple.py 0 20 100 1 True 'MetaWorld_04_Top+Corner+Corner2+Corner3+BehindGripper.zip'
	python3 Simulation_TFM_Window.py 0 20 100 1 True 'MetaWorld_0411_ALL'
	
TFM_EXTRAS:
	python3 Simulation_TFM_Window_Cuadruple.py 0 20 200 1 True 'MetaWorld_054_Top+Corner+Corner2+BehindGripper.zip'
	python3 Simulation_TFM_Window_Quintuple.py 0 20 200 1 True 'MetaWorld_04_Top+Corner+Corner2+Corner3+BehindGripper.zip'
	python3 Simulation_TFM_Window.py 0 20 200 1 True 'MetaWorld_0411_ALL'

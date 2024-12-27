
# start_t = time.monotonic()
#     start_l = sensors_reader.get_reading().front_distance
#     prev_t = start_t
#     prev_l = start_l
#     prev_v = 0
#     has_started = False
#     while True:

#         l = sensors_reader.get_reading().front_distance
#         if prev_l == l:
#             continue
#         t = time.monotonic()
#         dt = prev_t - t
#         dl = l - prev_l
        
#         print(l)
        
#         v = (dl/1000)/dt
#         print(f'Velocity: {v} m/s')
#         if (v > 0.05 and not has_started):
#             start_t = t
#             start_l = l
#             has_started = True
            
#         if (v > 0.7):
#             print(f'Acceleration took {start_t-t}s and {start_l-l} mm')
        
#         prev_l = l
#         prev_t = t
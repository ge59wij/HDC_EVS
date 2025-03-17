import threading
import time

def watch_dog():
    time.sleep( 10800)  # If main script is still running after x s, exit
    print("Script hung, exiting...")
    exit(1)

watchdog = threading.Thread(target=watch_dog, daemon=True)
watchdog.start()

# Run your script here
exec(open("/space/chair-nas/tosy/PycharmProjects/Thesis_Implementation/MAIN/DAT_approach/grasphd_appraoch/with_pickle/main_enc.py").read())

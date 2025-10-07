from rp import *
from fire import Fire
def main(video:str,output_folder=None):
    # 2025-10-06 13:47:57.610418
    #Ryan Burgert 2024
    #Run this in a Jupyter notebook code cell for a realtime preview!
    
    #Setup:
    #    Run this in a Jupyter Notebook on a computer with at least one GPU
    #        `sudo apt install ffmpeg git`
    #        `pip install rp`
    #    The first time you run this it might be a bit slow (it will download necessary models)
    #    The `rp` package will take care of installing the rest of the python packages for you
    
    import rp
    rp.git_import('CommonSource') #If missing, installs code from https://github.com/RyannDaGreat/CommonSource
    import rp.git.CommonSource.noise_warp as nw
    
    FRAME = 2**-1 #We immediately resize the input frames by this factor, before calculating optical flow
                  #The flow is calulated at (input size) Ã FRAME resolution.
                  #Higher FLOW values result in slower optical flow calculation and higher intermediate noise resolution
                  #Larger is not always better - watch the preview in Jupyter to see if it looks good!
    FLOW = 2**4   #Then, we use bilinear interpolation to upscale the flow by this factor
                  #We warp the noise at (input size) Ã FRAME Ã FLOW resolution
                  #The noise is then downsampled back to (input size)
                  #Higher FLOW values result in more temporally consistent noise warping at the cost of higher VRAM usage and slower inference time
    LATENT = 8    #We further downsample the outputs by this amount - because 8 pixels wide corresponds to one latent wide in Stable Diffusion
                  #The final output size is (input size) Ã· LATENT regardless of FRAME and FLOW
    
    #LATENT = 2    #Uncomment this line for a prettier visualization! But for any Stable-Diffusion based model, use LATENT=8
    
    #You can also use video files or URLs
    #images = "doggy_butterfly.mp4"
    images=video
    title=get_file_name(images,include_file_extension=False)
    
    h,w=480,832
    output_folder=output_folder or f"cut_and_drag_{title}_{h}x{w}"
    
    fansi_print(f'DOING WARPED NOISE ON {video} --> {output_folder}','green cyan italic bold')
    if folder_exists(output_folder):
        print('SKIPPING BECAUSE IT EXISTS: ',output_folder)
        return
    
    images = load_video(images)
    images=resize_images(images,size=(h,w))
    #See this function's docstring for more information!
    output = nw.get_noise_from_video(
        images,
        remove_background=False, #Set this to True to matte the foreground - and force the background to have no flow
        visualize=True,          #Generates nice visualization videos and previews in Jupyter notebook
        save_files=True,         #Set this to False if you just want the noises without saving to a numpy file
        
        noise_channels=16,
        output_folder=output_folder,
        resize_frames=FRAME,
        resize_flow=FLOW,
        downscale_factor=round(FRAME * FLOW) * LATENT,
    );
    
    with SetCurrentDirectoryTemporarily(output.output_folder):
        save_image(images[0],'first_frame.png')
        save_video_mp4(images,'input_video.mp4',framerate=20)
    
    print("Noise shape:"  ,output.numpy_noises.shape)
    print("Flow shape:"   ,output.numpy_flows .shape)
    print("Output folder:",output.output_folder)
    
if __name__ == "__main__":
    Fire(main)

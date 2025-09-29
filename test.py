from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models.pix2pix_model import Pix2PixModel
import torchvision.transforms as transforms

if __name__ == "__main__":
    opt = TestOptions().parse()

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model: Pix2PixModel = Pix2PixModel(opt)
    model.setup(opt)

    # create a website
    results_dir = Path(opt.dataroot) / opt.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results

        output_tensors = visuals['output_generator']
        image_names = data['name']
        
        # Ensure image_names is a list, even for batch_size=1
        if isinstance(image_names, str):
            image_names = [image_names]
        
        # Convert tensor to PIL Image and save each image
        to_pil = transforms.ToPILImage()
        for j in range(output_tensors.shape[0]):  # Iterate over batch
            output_tensor = output_tensors[j]  # Get single image tensor
            image_name = image_names[j]  # Get corresponding name
            
            # Assuming output_tensor is in range [-1, 1], normalize to [0, 1]
            output_tensor = (output_tensor + 1) / 2  # Normalize to [0, 1]
            output_image = to_pil(output_tensor.clamp(0, 1))  # Convert to PIL Image
            
            # Save the image with the input name in results_dir
            output_path = results_dir / f"{image_name}.png"
            output_image.save(output_path)
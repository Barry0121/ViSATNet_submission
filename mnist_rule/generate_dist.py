"""Plot statistics on bit value of SATNet hidden layers"""
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from models import *
from utils.get_dataloader import get_test_loader
import numpy as np
import warnings
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_args():
    parser = argparse.ArgumentParser()
    # specification
    parser.add_argument('--mode', type=str, default='config_distribution',
                        choices=['config_distribution', 'bits_distribution',
                                 'visualize_patches', 'extract_dataset', 'verify_patches'])
    parser.add_argument('--sampling', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument("--save_raw", type=bool, default=True)
    parser.add_argument("--save_formatted", type=bool, default=True)
    parser.add_argument('--k', type=int, default=20)

    # specify trained model
    parser.add_argument('--model', type=str, default='hrclf')
    parser.add_argument('--m', type=int, default=200)
    parser.add_argument('--aux', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2.e-3)
    parser.add_argument('--stride', type=int, default=7)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--threshold', type=float, default=0.5)

    return parser.parse_args()


def _get_num_patches(stride):
    return 784 // (stride ** 2)


def _get_model_dataset(args, device):
    exp_name = _get_model_path(args)
    test_loader = get_test_loader(args.dataset, os.path.join(args.root_dir, args.dataset), args.batch_size,
                                  device=device)
    model = HierarchicalClassifier(
        m=args.m, aux=args.aux, stride=args.stride, hidden_dim=args.hidden_dim)
    state_path = os.path.join(exp_name, 'model', 'best_model.pt')
    checkpoint = torch.load(state_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device), test_loader


def _get_model_path(args):
    """Return the target model/loss path given arguments.
    If the path doesn't exist, make it exist."""
    exp_name = os.path.join(args.save_dir, args.dataset,
                            (f"model_{args.model}_m_{args.m}_aux_{args.aux}"
                             f"_dim_{args.hidden_dim}_stride_{args.stride}"
                             f"_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}"))
    os.makedirs(exp_name, exist_ok=True)
    return exp_name


def _make_directory(args, config=True):
    exp_name = _get_model_path(args)
    num_classes = 10
    num_patches = _get_num_patches(args.stride)

    # Two different modes
    if config:
        # Check if path exists already
        if os.path.exists(os.path.join(str(exp_name), "digits", f"digit=0")):
            return
        else:
            for c in range(num_classes):
                os.makedirs(
                    os.path.join(str(exp_name), "digits", f"digit={c}"), exist_ok=True)
    else:
        # Check if path exists already
        if os.path.exists(os.path.join(str(exp_name), "digits", f"digit=0", f"patch=0")):
            return
        else:
            for c in range(num_classes):
                for p in range(num_patches):
                    os.makedirs(os.path.join(str(exp_name), "digits",
                                f"digit={c}", f"patch={p}"), exist_ok=True)


def _save_bits(args, digit, patch, bit, bits_arr, correctness_arr):
    """
    Save bits in the .npz format. It can be retrieved with _load_bits.
    """
    exp_name = _get_model_path(args)
    _make_directory(args, config=False)
    save_fp = os.path.join(str(exp_name), "digits",
                           f"digit={digit}", f"patch={patch}", f"bit={bit}.npz")
    np.savez(save_fp, bit=bits_arr, correctness=correctness_arr)


def _load_bits(args, digit, patch, bit):
    """
    Retrieve the bits and correctness by digit and patch number.
    :return: bits ndarray and correctness ndarray.
    """
    exp_name = _get_model_path(args)
    save_fp = os.path.join(str(exp_name), "digits",
                           f"digit={digit}", f"patch={patch}", f"bit={bit}.npz")
    data = np.load(save_fp)
    return data['bit'], data['correctness']


def collect_results(args, device, save=True):
    """
    Evaluate model results against test dataset, collect the hidden bits (data_size, num_patches, num_bits), labels,
    and prediction correctness.

    :return: bits, correctness, labels
    """
    # Gather metadata for files
    exp_name = _get_model_path(args=args)
    save_fp = str(os.path.join(str(exp_name), 'raw_data.npz'))

    # Determine to run evaluation or not
    if not os.path.exists(save_fp):
        # Load model and test dataset
        print("\nNo existing files found, running evaluation...")
        model, dataloader = _get_model_dataset(args=args, device=device)
        model.eval()
        num_patches = _get_num_patches(args.stride)
        bits, correctness, labels, patches = None, None, None, None

        # Run evaluation
        print("\nGenearte bits for testing images...")
        loader = tqdm(dataloader)
        with torch.no_grad():
            for image, label in loader:
                image = image.to(device)
                label = label.to(device)
                # Generate output
                pred = model(image)
                # (batch_size, (784/stride**2)*bit_size)
                bit = model.hidden_act
                # (batch_size, (784/stride**2), bit_size)
                bit = bit.reshape(bit.size(0), num_patches, -1)
                pred = torch.argmax(pred, dim=-1)
                # 1=correct, 0=false; (batch_size, 1)
                correct = torch.where(pred == label, 1., 0.)
                # Get patches
                patch = image.squeeze(1).cpu()\
                    .unfold(1, args.stride, args.stride)\
                    .unfold(2, args.stride, args.stride)\
                    .contiguous().view(image.size(0), -1, args.stride*args.stride)

                # Concat batched results in list
                if bits is None:
                    bits, correctness, labels, patches = bit, correct, label, patch
                else:
                    bits = torch.cat([bits, bit], dim=0)
                    correctness = torch.cat([correctness, correct], dim=0)
                    labels = torch.cat([labels, label], dim=0)
                    patches = torch.cat([patches, patch], dim=0)

        # Reformat collected results from list to numpy array
        labels = labels.cpu().numpy()
        correctness = correctness.cpu().numpy()
        bits = bits.cpu().numpy()
        patches = patches.cpu().numpy()

        # Save the result
        print("Saving raw results...")
        if save:
            np.savez(save_fp, bits=bits, labels=labels,
                     correctness=correctness, patches=patches)

    # Attempt to retrieve data from previously saved files
    else:
        print("Found existing raw data file! Using data from the file.")
        data = np.load(save_fp)
        bits, correctness, labels, patches = data['bits'], data['correctness'], data['labels'], data['patches']

        # Return results
    print("Completed!\n")
    return bits, correctness, labels, patches

# =================== Plot Bits Distribution ===================


def format_bits(bits, correctness, labels, save=True):
    """
    Given the collected raw result (either as arguments or as a file path), categorize the bits by their truth label
    and stored with the correctness of model's prediction.
    :param bits: numpy.ndarray, the collected raw bits. (num_images, num_patch, bit_size)
    :param correctness: numpy.ndarray, the correctness of corresponding correctness. (num_images, boolean_int)
    :param labels: numpy.ndarray, the labels from 0-9. (num_images, label_int)
    :param save: boolean, whether to save this or not.
    :return: 2D dictionary, record has nested indices {(digit,patch,bit): ([],[])}.
    """
    print("\nFormatting the raw bits.")

    # Setup temporary storage
    num_patches = _get_num_patches(args.stride)
    record = {(d, p, b): [[], []]  # ([bit values], [correctness])
              # index by bits_array.shape[2]: num_bits
              for b in range(args.hidden_dim)
              # index by bits.shape[1]: num_patches
              for p in range(num_patches)
              for d in range(0, 10)}  # indexed by labels which is indexed by bits.shape[0]

    # Obtain shared indices and sort them according to their label
    # Runtime could be an issue, but let's ignore for now
    for pi in range(bits.shape[1]):
        for bi in range(bits.shape[2]):
            for ii in range(bits.shape[0]):
                record[(labels[ii], pi, bi)][0].append(bits[ii, pi, bi])
                record[(labels[ii], pi, bi)][1].append(correctness[ii])

    # Reformatting the values
    print(f"Saving results...")
    for d, p, b in tqdm(record):
        # Save by the digits, patches, bits index
        if save:
            _save_bits(args, digit=d, patch=p, bit=b,
                       bits_arr=np.array(record[(d, p, b)][0]),
                       correctness_arr=np.array(record[(d, p, b)][1]))

    print("Everything is saved!\n")


def _plot_one_bit(args, d, p, b):
    # Load target bits data
    exp_name = _get_model_path(args)
    image_path = os.path.join(str(exp_name), "digits",
                              f"digit={d}", f"patch={p}", f"bit={b}.png")
    bits, _ = _load_bits(args=args, digit=d, patch=p, bit=b)

    # Plot a histogram plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"bit={b} at patch={p}, digit={d}")
    ax.set_xlim(0.0, 1.0)
    sns.histplot(data={'bits': bits}, x='bits', stat='density')
    plt.savefig(image_path, format='png')
    plt.close(fig)


def plot_distribution(args):
    """
    Populate the digits folder with histogram plots of the bits.
    """
    # Get the digit folder location and traverse through the bits.
    for d in tqdm(range(0, 10), desc="Digits", position=0):
        for p in tqdm(range(_get_num_patches(args.stride)), desc="Patches", position=1, leave=False):
            for b in tqdm(range(args.hidden_dim), desc="Bits", position=2, leave=False):
                _plot_one_bit(args, d, p, b)

    print("Complete, all figures saved! \n")

# =================== Plot Config Distribution ===================


def _save_config(args, digit, config_dict):
    """
    Save the distribution of configurations labeled by patch indices.
    """
    exp_name = _get_model_path(args)
    _make_directory(args, config=True)
    save_fp = os.path.join(str(exp_name), "digits",
                           f"digit={digit}", "config_dist.npz")
    np.savez(save_fp, **config_dict)


def _load_config(args, digit):
    """
    Load the distribution against bits configurations dictionary.
    :return: dict, keys are different bit configurations (ex: 0010 if hidden_dim=4).
    """
    exp_name = _get_model_path(args)
    save_fp = os.path.join(str(exp_name), "digits",
                           f"digit={digit}", f"config_dist.npz")
    data = np.load(save_fp)
    return data


def format_config(bits, correctness, labels, save=True):
    """
    Given the collected raw result (either as arguments or as a file path), categorize the bits by their truth label
    and stored with the correctness of model's prediction.
    :param bits: numpy.ndarray, the collected raw bits. (num_images, num_patch, bit_size)
    :param correctness: numpy.ndarray, the correctness of corresponding correctness. (num_images, boolean_int)
    :param labels: numpy.ndarray, the labels from 0-9. (num_images, label_int)
    :param save: boolean, whether to save this or not.
    :return: 2D dictionary, record has nested indices {(digit,patch,bit): ([],[])}.
    """
    print("\nFormatting bits by possible configurations.")

    # Setup temporary storage
    config = []
    for integer in range(2**args.hidden_dim):
        binary = bin(integer)[2:]

        # Apply padding
        if len(binary) < args.hidden_dim:
            binary = str((args.hidden_dim-len(binary))*'0'+binary)

        config.append(binary)

    record = {d: {c: [] for c in config} for d in range(10)}

    # Threshold all the bis values
    threshold = args.threshold
    binary_bits = np.where(bits < threshold, 0, 1)  # (images, patches, bits)
    sparse_measure = np.where((bits > threshold) & (bits < 1-threshold), 1, 0)
    print("About {:.3f}% of the bits are in between {} and {}."
          .format((np.sum(sparse_measure)/sparse_measure.size)*100,
                  threshold,
                  1-threshold))

    # Populate 'record' with a list of patch numbers
    for i, d in enumerate(labels):
        img = binary_bits[i]
        for p, bits in enumerate(img):
            record[d][''.join(bits.astype(str))].append(p)

    # Save the record by digits
    if save:
        for d, dist in record.items():
            _save_config(args, digit=d, config_dict=dist)
    print("Everything is saved!\n")


def _plot_one_config(args, digit):
    """
    Plot one distribution by configurations given the digit.
    """
    # Get the distribution
    config_dict = dict(_load_config(args, digit=digit))

    # Sort and plot histogram for top-k configurations
    topk_record = sorted(
        config_dict.items(), key=lambda x: len(x[1]), reverse=True)[:args.k]
    topk_record = dict(topk_record)

    keys = list(topk_record.keys())
    values = [len(v) for v in topk_record.values()]

    # Save path
    exp_name = _get_model_path(args)
    save_fp = os.path.join(str(exp_name), "digits",
                           f"digit={digit}", f"config_dist.png")

    # Plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(25, 15))
    ax.set_title(f"Digit={digit}")
    ax.set_xlabel("Configurations")
    ax.set_ylabel("Number of patches")
    sns.barplot(x=keys, y=values)
    plt.xticks(rotation=90, fontsize=5)
    plt.savefig(save_fp, format='png', dpi=300)


def plot_configuration_distributions(args):
    """
    Populate the digits folder with histogram plots of the bits.
    """
    print("\nPlotting configuration distributions...")

    # 1) Plot by digits, get the digit folder location and traverse through the bits.
    for d in tqdm(range(0, 10), desc="Digits", position=0):
        _plot_one_config(args, d)

    # 2) Plot an overall distribution
    aggregate_dict = {}
    for d in tqdm(range(10), desc="Digits"):
        if d == 0:
            aggregate_dict = {config: len(
                patches) for config, patches in _load_config(args, digit=d).items()}
        else:
            for config, patches in _load_config(args, digit=d).items():
                aggregate_dict[config] += len(patches)

    # Sort and plot histogram for top-k configurations
    topk_record = sorted(
        aggregate_dict.items(), key=lambda x: x[1], reverse=True)[:args.k]
    topk_record = dict(topk_record)

    keys = list(topk_record.keys())
    values = list(topk_record.values())

    exp_name = _get_model_path(args)
    save_fp = os.path.join(str(exp_name), f"overall_config_dist.png")

    fig, ax = plt.subplots(1, 1, figsize=(25, 15))
    ax.set_title(f"Overall")
    ax.set_xlabel("Configurations")
    ax.set_ylabel("Number of patches")
    sns.barplot(x=keys, y=values)
    # rotate x labels for better viewing
    plt.xticks(rotation=90, fontsize=5)
    plt.savefig(save_fp, format='png', dpi=300)

    print("Complete, all figures saved! \n")

# ================= Visualize Patches =====================


def visualize_patches(args, bits, patches, k=20, sample=False, sample_size=100, save=True):
    """
    Visualize patches either by global averaging across images or averaging sampled instances.
    """
    print("\nGenerating Patches Visualization...")
    # Reterive patches and bits by patches
    bits = bits.reshape(-1, args.hidden_dim)
    patches = patches.reshape(-1, args.stride**2)

    # Setup temporary storage
    config = []
    for integer in range(2**args.hidden_dim):
        binary = bin(integer)[2:]

        # Apply padding
        if len(binary) < args.hidden_dim:
            binary = str((args.hidden_dim-len(binary))*'0'+binary)

        config.append(binary)

    record = {c: [] for c in config}

    # Threshold all the bis values
    threshold = args.threshold
    binary_bits = np.where(bits < threshold, 0, 1)  # (images, patches, bits)
    sparse_measure = np.where((bits > threshold) & (bits < 1-threshold), 1, 0)
    print("About {:.3f}% of the bits are in between {} and {}."
          .format((np.sum(sparse_measure)/sparse_measure.size)*100,
                  threshold,
                  1-threshold))

    # Populate 'record' with a list of patch numbers
    for i, bit in enumerate(binary_bits):
        record[''.join(bit.astype(str))].append(patches[i])

    # Select the top k configurations
    topk_record = sorted(
        record.items(), key=lambda x: len(x[1]), reverse=True)[:k]
    topk_record = dict(topk_record)

    # Obtain the maximum amount we can sample from all configuration
    for k, v in topk_record.items():
        if len(v) < sample_size and len(v) != 0:
            sample_size = len(v)

    # Obtain the average of sampled patches
    for k, v in topk_record.items():
        if len(v) == 0:
            print(f"Failed to find any patches for {k}")
            continue
        if sample:
            sample_index = np.random.choice(
                range(len(v)), size=sample_size, replace=False).astype(int)

            if sample_size < 2:
                _visualize_one_patch(args, configuration=k,
                                     pixels=v[sample_index[0]])
            else:
                _visualize_one_patch(args, configuration=k,
                                     pixels=np.mean(v[sample_index[0]], axis=0))
        else:
            if len(v) < 2:
                _visualize_one_patch(args, configuration=k, pixels=v)
            else:
                _visualize_one_patch(args, configuration=k,
                                     pixels=np.mean(v, axis=0))

    print("Finished! Everything is saved under output.\n")


def _visualize_one_patch(args, configuration, pixels):
    """
    Visualize a specified patch and save it at designated location.
    """
    exp_name = _get_model_path(args)
    _make_directory(args, config=True)
    os.makedirs(os.path.join(str(exp_name), "digits",
                f"top{args.k}_configs"), exist_ok=True)
    save_fp = os.path.join(str(exp_name), "digits", f"top{args.k}_configs",
                           f"configuration={configuration}.png")

    pixels = np.array(pixels).reshape(
        args.stride, args.stride)
    plt.imsave(save_fp, pixels, cmap='grey', vmin=0, vmax=1)

# ================= Extract Top-k Dataset =====================


def extact_dataset(args, bits, patches, k=20):
    """
    Save ready-to-use dataset of top-k most populated configurations (default k=20).
    """
    # Reterive patches and bits by patches
    bits = bits.reshape(-1, args.hidden_dim)
    patches = patches.reshape(-1, args.stride**2)

    # Setup temporary storage
    config = []
    for integer in range(2**args.hidden_dim):
        binary = bin(integer)[2:]

        # Apply padding
        if len(binary) < args.hidden_dim:
            binary = str((args.hidden_dim-len(binary))*'0'+binary)

        config.append(binary)

    record = {c: [] for c in config}

    # Threshold all the bis values
    binary_bits = np.where(bits < args.threshold, 0,
                           1)  # (images, patches, bits)
    sparse_measure = np.where((bits > args.threshold)
                              & (bits < 1-args.threshold), 1, 0)
    print("About {:.3f}% of the bits are in between {} and {}."
          .format((np.sum(sparse_measure)/sparse_measure.size)*100,
                  args.threshold,
                  1-args.threshold))

    # Populate 'record' with a list of patch numbers

    for i, bit in enumerate(binary_bits):
        record[''.join(bit.astype(str))].append(patches[i])

    # Select the top k configurations
    topk_record = sorted(
        record.items(), key=lambda x: len(x[1]), reverse=True)[:k]
    topk_record = dict(topk_record)

    # Save top k
    exp_name = _get_model_path(args)
    save_fp = os.path.join(str(exp_name), "patch_config_data.pickle")

    # Save file
    with open(save_fp, 'wb') as handle:
        pickle.dump(topk_record, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return topk_record

# =========== Verify patches ============


def verify_patches(args, bits, patches, correctness, labels):
    print("\nCreating different version of a randomly sampled image.")
    avg_patches_config = _generate_avg_patches(args, bits, patches)

    config_from_image, patches_from_image, correctness, labels = _find_image_config_and_patches(
        args, bits, patches, correctness, labels)

    images_with_swapped_patches = _map_image_patches_to_config_avg(
        patches_from_image, config_from_image, avg_patches_config)

    # Save the sampled image without the change
    save_path_root = os.path.join(
        str(_get_model_path(args)), "digits", "switch_patches")
    os.makedirs(save_path_root, exist_ok=True)

    # Fold the patches back to an image
    # np.expand_dim = torch.unsqueeze; use this to recreate the batch dimension.
    # Desired input dimension should be (batch, patch_size**2, patch_numbers)
    original_image = F.fold(torch.from_numpy(np.expand_dims(patches_from_image, axis=0)).permute(0, 2, 1),
                            output_size=(28, 28), kernel_size=args.stride, stride=args.stride).squeeze().numpy()


    print(patches_from_image.shape)


    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imsave(os.path.join(save_path_root, "original.png"),
               original_image, cmap='gray', vmin=0, vmax=1)
    plt.close()

    # Plot patches swapped images: images with single patch swap
    for i, image in enumerate(images_with_swapped_patches[:-1]):
        image = F.fold(torch.from_numpy(np.expand_dims(image, axis=0)).permute(0, 2, 1),
                       output_size=(28, 28), kernel_size=args.stride, stride=args.stride).squeeze().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imsave(os.path.join(save_path_root, f"switch_#{i}.png"),
                   image, cmap='gray', vmin=0, vmax=1)
        plt.close()

    # Plot last image: concatenation of all the patches by configuration
    image = images_with_swapped_patches[-1]
    image = F.fold(torch.from_numpy(np.expand_dims(image, axis=0)).permute(0, 2, 1),
                   output_size=(28, 28), kernel_size=args.stride, stride=args.stride).squeeze().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imsave(os.path.join(save_path_root, f"all_config_patches.png"),
               image, cmap='gray', vmin=0, vmax=1)
    plt.close()

    print("Finished image generation.\n")


def _generate_avg_patches(args, bits, patches):
    """
    Return the average patches in respect to all configurations in a dictionary.
    """
    # Reterive patches and bits by patches
    bits = bits.reshape(-1, args.hidden_dim)
    patches = patches.reshape(-1, args.stride**2)

    # Setup temporary storage
    config = []
    for integer in range(2**args.hidden_dim):
        binary = bin(integer)[2:]

        # Apply padding
        if len(binary) < args.hidden_dim:
            binary = str((args.hidden_dim-len(binary))*'0'+binary)

        config.append(binary)

    record = {c: [] for c in config}

    # Threshold all the bis values
    binary_bits = np.where(bits < args.threshold, 0,
                           1)  # (images, patches, bits)

    # Populate 'record' with a list of patch values
    for i, bit in enumerate(binary_bits):
        binary_bits_str = ''.join(bit.astype(str))
        record[binary_bits_str].append(patches[i])

    # Apply averaging
    for config, patches in record.items():
        record[config] = np.mean(record[config], axis=0)

    return record


def _find_image_config_and_patches(args, bits, patches, correctness, labels, sample_size=1):
    """
    Return one image's patches and bits through sampling.
    """
    # Sample 1 image's bits and patches
    sample_indx = np.random.choice(range(bits.shape[0]), size=sample_size)
    sampled_bits, sampled_patches = bits[sample_indx].squeeze(
    ), patches[sample_indx].squeeze()
    sampled_correctness, sampled_labels = correctness[sample_indx].squeeze(
    ), labels[sample_indx].squeeze()

    # Threshold the bits to create configuration
    sampled_binary_bits = np.where(sampled_bits < args.threshold, 0, 1)

    # Populate 'record' with a list of patch values
    sampled_binary_bits = np.stack(
        [''.join(bit.astype(str)) for bit in sampled_binary_bits])

    return sampled_binary_bits, sampled_patches, sampled_correctness, sampled_labels


def _map_image_patches_to_config_avg(patches_from_image, config_from_image, avg_patches_config):
    """
    Map one patch from an image to its avg equivalence.
    """

    new_images, config_only_image = [], patches_from_image.copy()
    for patch_indx, config in enumerate(config_from_image):
        new_image = patches_from_image.copy()
        new_image[patch_indx] = avg_patches_config[config]
        config_only_image[patch_indx] = avg_patches_config[config]
        new_images.append(new_image)
    new_images.append(config_only_image)
    return np.stack(new_images, axis=0)


# ================ Main ==================
if __name__ == '__main__':
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    bits, correctness, labels, patches = collect_results(
        args, device, save=args.save_raw)

    match args.mode:
        case "config_distribution":
            format_config(bits, correctness, labels, save=args.save_formatted)
            plot_configuration_distributions(args)
        case 'bits_distribution':
            format_bits(bits, correctness, labels, save=args.save_formatted)
            plot_distribution(args=args)
        case 'visualize_patches':
            visualize_patches(args, bits, patches, sample=args.sampling)
        case 'extract_dataset':
            extact_dataset(args, bits, patches)
        case 'verify_patches':
            verify_patches(args, bits, patches, correctness, labels)
        case _:
            raise NameError("The specified mode is not available, "
                            "must be one of the following: [config_distribution, bits_distribution, visualize_patches]")

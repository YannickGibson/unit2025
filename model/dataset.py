"""This is a dataset file given to us during the hackathon. We did not create it."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
import xarray as xr
from torch.utils.data import Dataset

CHANNEL_DICT = {
    "red": "s2_B04",
    "green": "s2_B03",
    "blue": "s2_B02",
    "nir": "s2_B8A",
    "mask": "s2_dlmask",
    "ndvi": "ndvi",
    "evi": "evi",
    "class": "esawc_lc",
    "longitude": "lon",  # longitude
    "latitude": "lat",  # latitude
    "elevation_nasadem": "nasa_dem",  # elevation by NASADEM HGT v001
    "elevation_copernicus": "cop_dem",  # eleveation by Copernicus DEM GLO-30
    "elevation_alos": "alos_dem",  # elevation by ALOS World 3D-30m
}


ADDITIONAL_INFO_DICT = {
    "wind_speed": "eobs_fg",  # Mean wind speed in meters per second
    "humidity": "eobs_hu",  # Relative humidity in percentage
    "pressure": "eobs_pp",  # Mean sea level pressure in hPa
    "radiation": "eobs_qq",  # Amount of shortwave radiation in W/m^2
    "rainfall": "eobs_rr",  # Total precipitation in mm
    # Mean temperature of the air in degrees Celsius
    "air_temperature_mean": "eobs_tg",
    # Minimum temperature of the air in degrees Celsius
    "air_temperature_minimum": "eobs_tn",

}


class BaseGreenEarthNetDataset(Dataset):
    def __init__(self,
                 folder: Path | str,
                 input_channels: List[str],
                 target_channels: List[str],
                 additional_info_list: List[str] | None = None,
                 time: bool = False,
                 transform: T.Compose | None = None,
                 target_transform: T.Compose | None = None,
                 use_mask: bool = True):
        """Base dataset for UNIT competition. This dataset returns 
        a dictionary with the whole sequence of images and the target.

        Parameters:
        ----------
        folder : Path | str
            Path to the folder containing the dataset files.
        input_channels : List
            List of input channels to be used.
        target_channels : List
            List of target channels to be used.
        additional_info_list : List | None, optional
            List of additional information to be used. Defaults to None. 
        time : bool, optional
            Whether to return the time of the capture. Defaults to False.
        transform : T.Compose | None, optional
            Transformations to apply to the input data. Defaults to None.
        target_transform : T.Compose | None, optional
            Transformations to apply to the target data. Defaults to None.
        use_mask : bool, optional
            Whether to use a mask. If use_mask is set to True, invalid pixels are set to np.nan. Defaults to True.
        """

        self.files = list(Path(folder).rglob("*.nc"))
        self.transform = transform
        self.target_transform = target_transform
        self.use_mask = use_mask
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.additional_info_list = additional_info_list
        self.time = time

        self._length_sequence = 30

        self._check_channels(input_channels, "Input")
        self._check_channels(target_channels, "Target")
        self._check_additional_info(
            additional_info_list, "Additional Info") if additional_info_list else None

    def _check_channels(self, channels: List, channel_type: str):
        """Check if the channels are in the channel dictionary keys.

        Parameters:
        -----------
        channels : List
            List of channels to check.
        channel_type : str
            Type of channels being checked (e.g., "Input" or "Target").
        """
        for channel in channels:
            if channel not in CHANNEL_DICT.keys():
                raise ValueError(
                    f"{channel_type} channel '{channel}' is not in the channel dictionary keys. "
                    f"Use one of {', '.join(CHANNEL_DICT.keys())}.")

    def _check_additional_info(self, additional_info_list: List, additional_info_type: str):
        """Check if the additional information are in the additional info dictionary keys.

        Parameters:
        -----------
        additional_info_list : List
            List of additional information to check.
        additional_info_type : str
            Type of additional information being checked (e.g., "Additional Info").
        """
        for additional_info in additional_info_list:
            if additional_info not in ADDITIONAL_INFO_DICT.keys():
                raise ValueError(
                    f"{additional_info_type} '{additional_info}' is not in the additional info dictionary keys. "
                    f"Use one of {', '.join(ADDITIONAL_INFO_DICT.keys())}.")

    def _get_channel(self, minicube: xr.Dataset, channel_name: str, sequence_length: int):
        """Get the channel from the minicube DataArray."""
        channel_array = minicube[CHANNEL_DICT[channel_name]].values
        if channel_name in ["class", "longtitude", "latitude"] or "elevation" in channel_name:
            channel_array = np.repeat(
                channel_array[np.newaxis, :, :], sequence_length, axis=0)
        return channel_array

    def _get_additional_info(self, minicube: xr.Dataset, additional_info_name: str, sequence_length: int):
        """Get the additional information from the minicube DataArray."""
        additional_info_array = minicube[ADDITIONAL_INFO_DICT[additional_info_name]].values
        return additional_info_array

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        """Retrieves the input and target data for a given index from the dataset.

        Parameters:
        -----------
        idx : int
            Index of the data to retrieve.

        Returns:
        --------
        Dict[str, np.ndarray]: A dictionary containing:
            - 'inputs' (np.ndarray): The input data array in shape [L, CH_i, H, W], 
               where L is the sequence length and CH_i is the number of input channels.
            - 'targets' (np.ndarray): The target data array in shape [L, CH_t, H, W], 
               where L is the sequence length and CH_t is the number of target channels.
            - 'additional_info' (np.ndarray): The additional information array in shape [L, CH_a],
               where L is the sequence length and CH_a is the number of additional information channels. Optional.
            - 'time' (np.ndarray): The time of the capture. Optional.

        """

        file = self.files[idx]
        minicube = xr.open_dataset(file).isel(time=slice(4, None, 5))
        minicube = self.compute_ndvi(minicube)
        minicube = self.compute_evi(minicube)

        inputs = np.stack([self._get_channel(minicube, channel, self._length_sequence)
                          for channel in self.input_channels], axis=1)
        targets = np.stack([self._get_channel(minicube, channel, self._length_sequence)
                           for channel in self.target_channels], axis=1)

        if self.use_mask:
            mask = self._get_channel(minicube, "mask", self._length_sequence)
            mask = np.expand_dims(mask, axis=1)
            inputs = np.where(mask == 0, inputs, np.nan)
            targets = np.where(mask == 0, targets, np.nan)

        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        out = {
            'inputs': inputs,
            'targets': targets
        }

        if self.additional_info_list is not None:
            out['additional_info'] = np.stack(
                [self._get_additional_info(minicube, info, self._length_sequence) for info in self.additional_info_list], axis=1)

        if self.time:
            out['times'] = minicube.time.values

        return out

    def compute_ndvi(self, minicube):
        """Compute the Normalized Difference Vegetation Index (NDVI) for the given minicube.

        NDVI is calculated using the formula:
        NDVI = (NIR - Red) / (NIR + Red + 1e-8)

        Parameters:
        -----------
        minicube : xarray.Dataset)
            The input dataset containing the spectral bands.

        Returns:
        --------
        minicube: xarray.Dataset
            The input dataset with an additional 'ndvi' variable representing the NDVI.
        """
        minicube["ndvi"] = (minicube.s2_B8A - minicube.s2_B04) / \
            (minicube.s2_B8A + minicube.s2_B04 + 1e-8)
        return minicube

    def compute_evi(self, minicube):
        """Compute the Enhanced Vegetation Index (EVI) for the given minicube.

        EVI is calculated using the formula:
        EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))

        Parameters:
        -----------
        minicube : xarray.Dataset)
            The input dataset containing the spectral bands.

        Returns:
        --------
        minicube: xarray.Dataset
            The input dataset with an additional 'evi' variable representing the EVI.
        """
        minicube["evi"] = 2.5 * ((minicube.s2_B8A - minicube.s2_B04) /
                                 (minicube.s2_B8A + 6 * minicube.s2_B04 - 7.5 * minicube.s2_B02 + 1))
        return minicube



class SeqGreenEarthNetDataset(BaseGreenEarthNetDataset):
    def __init__(self,
                 folder: Path | str,
                 input_channels: List[str],
                 target_channels: List[str],
                 additional_info_list: List[str] | None = None,
                 time: bool = False,
                 transform: T.Compose | None = None,
                 target_transform: T.Compose | None = None,
                 use_mask: bool = True,
                 return_filename: bool = False):
        """Base dataset for UNIT competition. This dataset returns 
        a dictionary with the whole sequence of images and the target.

        Parameters:
        ----------
        folder : Path | str
            Path to the folder containing the dataset files.
        input_channels : List
            List of input channels to be used.
        target_channels : List
            List of target channels to be used.
        additional_info_list : List | None, optional
            List of additional information to be used. Defaults to None. 
        time : bool, optional
            Whether to return the time of the capture. Defaults to False.
        transform : T.Compose | None, optional
            Transformations to apply to the input data. Defaults to None.
        target_transform : T.Compose | None, optional
            Transformations to apply to the target data. Defaults to None.
        use_mask : bool, optional
            Whether to use a mask. If use_mask is set to True, invalid pixels are set to np.nan. Defaults to True.
        return_filename : bool, optional
            Whether to return the filename. Defaults to False.
        """

        self.files = list(Path(folder).glob("*.npz"))
        self.transform = transform
        self.target_transform = target_transform
        self.use_mask = use_mask
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.additional_info_list = additional_info_list
        self.time = time
        self.return_filename = return_filename

        self._check_channels(input_channels, "Input")
        self._check_channels(target_channels, "Target")
        self._check_additional_info(
            additional_info_list, "Additional Info") if additional_info_list else None

    def _get_channel(self, data: dict, channel_name: str):
        """Get the channel from the dict."""
        channel_array = data[channel_name]
        return channel_array

    def _get_additional_info(self, data: dict, additional_info_name: str):
        """Get the additional information from the minicube DataArray."""
        additional_info_array = data[additional_info_name]
        return additional_info_array

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        """For given index it retrives 5 consecutive images with time gap of 5 days between them 
        and the target image have a time gap of 25 days from the last image in the input sequence.

        Parameters:
        -----------
        idx : int
            Index of the data to retrieve.

        Returns:
        --------
        Dict[str, np.ndarray]: A dictionary containing:
            - 'inputs' (np.ndarray): The input data array in shape [L, CH_i, H, W], 
               where L is the sequence length and CH_i is the number of input channels.
            - 'targets' (np.ndarray): The target data array in shape [L, CH_t, H, W], 
               where L is the sequence length and CH_t is the number of target channels.
            - 'additional_info' (np.ndarray): The additional information array in shape [L, CH_a],
               where L is the sequence length and CH_a is the number of additional information channels. Optional.
            - 'time' (np.ndarray): The time of the capture. Optional.

        """

        file = self.files[idx]
        data = np.load(file, allow_pickle=True)

        data_input_channels = data["inputs"].item()
        data_target_channels = data["targets"].item()

        inputs = np.stack([self._get_channel(data_input_channels, channel)
                          for channel in self.input_channels], axis=1)

        targets = np.stack([self._get_channel(data_target_channels, channel)
                           for channel in self.target_channels], axis=1)

        if self.use_mask:
            mask = self._get_channel(data_input_channels, "mask")
            mask = np.expand_dims(mask, axis=1)
            inputs = np.where(mask == 0, inputs, np.nan)

            mask = self._get_channel(data_target_channels, "mask")
            mask = np.expand_dims(mask, axis=1)
            targets = np.where(mask == 0, targets, np.nan)

        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        out = {
            'inputs': inputs,
            'targets': targets,
        }

        if self.additional_info_list is not None:

            data_input_additional_info = data["input_additional_info"].item()
            data_target_additional_info = data["target_additional_info"].item()

            out['input_additional_info'] = np.stack(
                [self._get_additional_info(data_input_additional_info, info) for info in self.additional_info_list], axis=1)

            out['target_additional_info'] = np.stack(
                [self._get_additional_info(data_target_additional_info, info) for info in self.additional_info_list], axis=1)

        if self.time:
            out['input_times'] = data["input_times"]
            out['target_times'] = data["target_times"]

        if self.return_filename:
            out['filename'] = file

        return out
    


 

class BaseGreenEarthNetDatasetWithFillingOption(BaseGreenEarthNetDataset):
    def __init__(self,
                 folder: Path | str,
                 input_channels: List[str],
                 target_channels: List[str],
                 additional_info_list: List[str] | None = None,
                 time: bool = False,
                 transform: T.Compose | None = None,
                 target_transform: T.Compose | None = None,
                 use_mask: bool = True,
                 use_fill_for_clouds: bool = True,
                 ):
        """Base dataset for UNIT competition. This dataset returns 
        a dictionary with the whole sequence of images and the target.

        Parameters:
        ----------
        folder : Path | str
            Path to the folder containing the dataset files.
        input_channels : List
            List of input channels to be used.
        target_channels : List
            List of target channels to be used.
        additional_info_list : List | None, optional
            List of additional information to be used. Defaults to None. 
        time : bool, optional
            Whether to return the time of the capture. Defaults to False.
        transform : T.Compose | None, optional
            Transformations to apply to the input data. Defaults to None.
        target_transform : T.Compose | None, optional
            Transformations to apply to the target data. Defaults to None.
        use_mask : bool, optional
            Whether to use a mask. If use_mask is set to True, invalid pixels are set to np.nan. Defaults to True.
        """

        self.files = list(Path(folder).rglob("*.nc"))
        self.transform = transform
        self.target_transform = target_transform
        self.use_mask = use_mask
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.additional_info_list = additional_info_list
        self.time = time
        self.use_fill_for_clouds = use_fill_for_clouds

        

        self._length_sequence = 30

        self._check_channels(input_channels, "Input")
        self._check_channels(target_channels, "Target")
        self._check_additional_info(
            additional_info_list, "Additional Info") if additional_info_list else None

    def _check_channels(self, channels: List, channel_type: str):
        """Check if the channels are in the channel dictionary keys.

        Parameters:
        -----------
        channels : List
            List of channels to check.
        channel_type : str
            Type of channels being checked (e.g., "Input" or "Target").
        """
        for channel in channels:
            if channel not in CHANNEL_DICT.keys():
                raise ValueError(
                    f"{channel_type} channel '{channel}' is not in the channel dictionary keys. "
                    f"Use one of {', '.join(CHANNEL_DICT.keys())}.")

    def _check_additional_info(self, additional_info_list: List, additional_info_type: str):
        """Check if the additional information are in the additional info dictionary keys.

        Parameters:
        -----------
        additional_info_list : List
            List of additional information to check.
        additional_info_type : str
            Type of additional information being checked (e.g., "Additional Info").
        """
        for additional_info in additional_info_list:
            if additional_info not in ADDITIONAL_INFO_DICT.keys():
                raise ValueError(
                    f"{additional_info_type} '{additional_info}' is not in the additional info dictionary keys. "
                    f"Use one of {', '.join(ADDITIONAL_INFO_DICT.keys())}.")

    def _get_channel(self, minicube: xr.Dataset, channel_name: str, sequence_length: int):
        """Get the channel from the minicube DataArray."""
        channel_array = minicube[CHANNEL_DICT[channel_name]].values
        if channel_name in ["class", "longtitude", "latitude"] or "elevation" in channel_name:
            channel_array = np.repeat(
                channel_array[np.newaxis, :, :], sequence_length, axis=0)
        return channel_array

    def _get_additional_info(self, minicube: xr.Dataset, additional_info_name: str, sequence_length: int):
        """Get the additional information from the minicube DataArray."""
        additional_info_array = minicube[ADDITIONAL_INFO_DICT[additional_info_name]].values
        return additional_info_array

    def __len__(self):
        return len(self.files)
 
    
    def fill_clouds_with_mean(self, inputs, minicube):
        """
        Fill cloud-masked pixels with mean values.
        Prioritizes mean from future frames, falls back to previous frames.
        
        Parameters:
        - inputs: Input array to be processed
        - minicube: Source of cloud mask
        - self: Instance with method to get channel
        
        Returns:
        - Processed inputs with cloud pixels filled
        """
        if self.use_fill_for_clouds:

            # Get cloud mask and ensure correct boolean shape
            cloud_mask = self._get_channel(minicube, "mask", self._length_sequence).astype(bool)
            cloud_mask = np.expand_dims(cloud_mask, axis=1)
            non_cloud_mask = ~cloud_mask
    
            # Try to calculate mean from future frames first
            try:
                future_mean_values = np.nanmean(np.where(non_cloud_mask[1:], inputs[1:], np.nan), axis=0, keepdims=True)
                
                # If future mean is available, use it
                if not np.all(np.isnan(future_mean_values)):
                    inputs = np.where(cloud_mask, future_mean_values, inputs)
                else:
                    # Fallback to previous frames if future mean is not available
                    previous_mean_values = np.nanmean(np.where(non_cloud_mask[:-1], inputs[:-1], np.nan), axis=0, keepdims=True)
                    inputs = np.where(cloud_mask, previous_mean_values, inputs)
            
            except IndexError:
                # If there are not enough frames for future mean, use previous frames
                previous_mean_values = np.nanmean(np.where(non_cloud_mask[:-1], inputs[:-1], np.nan), axis=0, keepdims=True)
                inputs = np.where(cloud_mask, previous_mean_values, inputs)

        return inputs
    
    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        """Retrieves the input and target data for a given index from the dataset.

        Parameters:
        -----------
        idx : int
            Index of the data to retrieve.

        Returns:
        --------
        Dict[str, np.ndarray]: A dictionary containing:
            - 'inputs' (np.ndarray): The input data array in shape [L, CH_i, H, W], 
               where L is the sequence length and CH_i is the number of input channels.
            - 'targets' (np.ndarray): The target data array in shape [L, CH_t, H, W], 
               where L is the sequence length and CH_t is the number of target channels.
            - 'additional_info' (np.ndarray): The additional information array in shape [L, CH_a],
               where L is the sequence length and CH_a is the number of additional information channels. Optional.
            - 'time' (np.ndarray): The time of the capture. Optional.

        """

        file = self.files[idx]
        minicube = xr.open_dataset(file).isel(time=slice(4, None, 5))
        minicube = self.compute_ndvi(minicube)
        minicube = self.compute_evi(minicube)

        inputs = np.stack([self._get_channel(minicube, channel, self._length_sequence)
                          for channel in self.input_channels], axis=1)
        targets = np.stack([self._get_channel(minicube, channel, self._length_sequence)
                           for channel in self.target_channels], axis=1)

        if self.use_mask:
            mask = self._get_channel(minicube, "mask", self._length_sequence)
            mask = np.expand_dims(mask, axis=1)
            inputs = np.where(mask == 0, inputs, np.nan)
            targets = np.where(mask == 0, targets, np.nan)


        
        if self.use_fill_for_clouds:
            inputs = self.fill_clouds_with_mean(inputs, minicube)
        
        
        if self.transform is not None:
            print( inputs)
            print("self.transform ")
            inputs = self.transform(inputs)
        
            if isinstance(inputs, torch.Tensor):
                print("isinstance.transform ")
                
                inputs = inputs.numpy()  # Convert back to NumPy if needed for visualization
        
            print("Transformed input shape:", inputs.shape)  # Debugging step

            print( inputs)
        """
        if self.transform is not None:
            print("self.transform ")
            inputs = self.transform(inputs)
"""
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        out = {
            'inputs': inputs,
            'targets': targets
        }

        if self.additional_info_list is not None:
            out['additional_info'] = np.stack(
                [self._get_additional_info(minicube, info, self._length_sequence) for info in self.additional_info_list], axis=1)

        if self.time:
            out['times'] = minicube.time.values


        return out
        
 
class SeqGreenEarthNetDatasetWithFillingOption(BaseGreenEarthNetDatasetWithFillingOption):
    def __init__(self,
                 folder: Path | str,
                 input_channels: List[str],
                 target_channels: List[str],
                 additional_info_list: List[str] | None = None,
                 time: bool = False,
                 transform: T.Compose | None = None,
                 target_transform: T.Compose | None = None,
                 use_mask: bool = True,
                 use_fill_for_clouds: bool = True,
                 return_filename: bool = False):
        """Base dataset for UNIT competition. This dataset returns 
        a dictionary with the whole sequence of images and the target.

        Parameters:
        ----------
        folder : Path | str
            Path to the folder containing the dataset files.
        input_channels : List
            List of input channels to be used.
        target_channels : List
            List of target channels to be used.
        additional_info_list : List | None, optional
            List of additional information to be used. Defaults to None. 
        time : bool, optional
            Whether to return the time of the capture. Defaults to False.
        transform : T.Compose | None, optional
            Transformations to apply to the input data. Defaults to None.
        target_transform : T.Compose | None, optional
            Transformations to apply to the target data. Defaults to None.
        use_mask : bool, optional
            Whether to use a mask. If use_mask is set to True, invalid pixels are set to np.nan. Defaults to True.
        return_filename : bool, optional
            Whether to return the filename. Defaults to False.
        """

        self.files = list(Path(folder).glob("*.npz"))
        self.transform = transform
        self.target_transform = target_transform
        self.use_mask = use_mask
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.additional_info_list = additional_info_list
        self.time = time
        self.return_filename = return_filename
        self.use_fill_for_clouds = use_fill_for_clouds
        
        self._length_sequence = 30
        self._check_channels(input_channels, "Input")
        self._check_channels(target_channels, "Target")
        self._check_additional_info(
            additional_info_list, "Additional Info") if additional_info_list else None

    def _get_channel(self, data: dict, channel_name: str):
        """Get the channel from the dict."""
        channel_array = data[channel_name]
        return channel_array

    def _get_channel_for_map(self, minicube: xr.Dataset, channel_name: str, sequence_length: int):
        """Get the channel from the minicube DataArray."""
        channel_array = minicube[CHANNEL_DICT[channel_name]].values
        if channel_name in ["class", "longtitude", "latitude"] or "elevation" in channel_name:
            channel_array = np.repeat(
                channel_array[np.newaxis, :, :], sequence_length, axis=0)
        return channel_array
    
    def _get_additional_info(self, data: dict, additional_info_name: str):
        """Get the additional information from the minicube DataArray."""
        additional_info_array = data[additional_info_name]
        return additional_info_array

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        """For given index it retrives 5 consecutive images with time gap of 5 days between them 
        and the target image have a time gap of 25 days from the last image in the input sequence.

        Parameters:
        -----------
        idx : int
            Index of the data to retrieve.

        Returns:
        --------
        Dict[str, np.ndarray]: A dictionary containing:
            - 'inputs' (np.ndarray): The input data array in shape [L, CH_i, H, W], 
               where L is the sequence length and CH_i is the number of input channels.
            - 'targets' (np.ndarray): The target data array in shape [L, CH_t, H, W], 
               where L is the sequence length and CH_t is the number of target channels.
            - 'additional_info' (np.ndarray): The additional information array in shape [L, CH_a],
               where L is the sequence length and CH_a is the number of additional information channels. Optional.
            - 'time' (np.ndarray): The time of the capture. Optional.

        """

        file = self.files[idx]
        data = np.load(file, allow_pickle=True)

        data_input_channels = data["inputs"].item()
        data_target_channels = data["targets"].item()

        inputs = np.stack([self._get_channel(data_input_channels, channel)
                          for channel in self.input_channels], axis=1)

        targets = np.stack([self._get_channel(data_target_channels, channel)
                           for channel in self.target_channels], axis=1)

        if self.use_mask:
            mask = self._get_channel(data_input_channels, "mask")
            mask = np.expand_dims(mask, axis=1)
            inputs = np.where(mask == 0, inputs, np.nan)

            mask = self._get_channel(data_target_channels, "mask")
            mask = np.expand_dims(mask, axis=1)
            targets = np.where(mask == 0, targets, np.nan)

        if self.use_fill_for_clouds:
            inputs = self.fill_clouds_with_mean(inputs, minicube)

        
        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        out = {
            'inputs': inputs,
            'targets': targets,
        }

        if self.additional_info_list is not None:

            data_input_additional_info = data["input_additional_info"].item()
            data_target_additional_info = data["target_additional_info"].item()

            out['input_additional_info'] = np.stack(
                [self._get_additional_info(data_input_additional_info, info) for info in self.additional_info_list], axis=1)

            out['target_additional_info'] = np.stack(
                [self._get_additional_info(data_target_additional_info, info) for info in self.additional_info_list], axis=1)

        if self.time:
            out['input_times'] = data["input_times"]
            out['target_times'] = data["target_times"]

        if self.return_filename:
            out['filename'] = file

        return out
    

 

    def compute_ndvi(self, minicube):
        """Compute the Normalized Difference Vegetation Index (NDVI) for the given minicube.

        NDVI is calculated using the formula:
        NDVI = (NIR - Red) / (NIR + Red + 1e-8)

        Parameters:
        -----------
        minicube : xarray.Dataset)
            The input dataset containing the spectral bands.

        Returns:
        --------
        minicube: xarray.Dataset
            The input dataset with an additional 'ndvi' variable representing the NDVI.
        """
        minicube["ndvi"] = (minicube.s2_B8A - minicube.s2_B04) / \
            (minicube.s2_B8A + minicube.s2_B04 + 1e-8)
        return minicube

    def compute_evi(self, minicube):
        """Compute the Enhanced Vegetation Index (EVI) for the given minicube.

        EVI is calculated using the formula:
        EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))

        Parameters:
        -----------
        minicube : xarray.Dataset)
            The input dataset containing the spectral bands.

        Returns:
        --------
        minicube: xarray.Dataset
            The input dataset with an additional 'evi' variable representing the EVI.
        """
        minicube["evi"] = 2.5 * ((minicube.s2_B8A - minicube.s2_B04) /
                                 (minicube.s2_B8A + 6 * minicube.s2_B04 - 7.5 * minicube.s2_B02 + 1))
        return minicube






def custom_collate_fn(batch, default_key=None):
    """Custom collate function for PyTorch DataLoader with support for datetime arrays."""
    elem = batch[0]

    if isinstance(elem, (torch.Tensor, np.ndarray)):
        return torch.stack([torch.from_numpy(b) if isinstance(b, np.ndarray) else b for b in batch], dim=0)

    elif isinstance(elem, dict):
        out = {}
        for key in elem:
            values = [d[key] for d in batch]

            if "times" in key:
                # Convert datetime arrays to pandas Series and extract time components
                values = [pd.Series(v) for v in values]
                out.update(custom_collate_fn(values, key))
            else:
                out[key] = custom_collate_fn(values)

        return out

    elif isinstance(elem, pd.Series):
        # Extract year, month, day from datetime Series
        return {
            f"{default_key}_years": torch.stack([torch.tensor(b.dt.year.values, dtype=torch.float32) for b in batch], dim=0),
            f"{default_key}_months": torch.stack([torch.tensor(b.dt.month.values, dtype=torch.float32) for b in batch], dim=0),
            f"{default_key}_days": torch.stack([torch.tensor(b.dt.day.values, dtype=torch.float32) for b in batch], dim=0)
        }

    else:
        raise TypeError(f"Unsupported data type: {type(elem)}")

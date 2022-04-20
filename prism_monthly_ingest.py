import argparse
from datetime import datetime
import ftplib
import logging
import os
import pprint
import re
import shutil
import tempfile
from typing import List
import zipfile

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

# ------------------------------------------------------------------------------
# FTP locations for the ASCII (point file) data at NCEI/NOAA
_PRISM_FTP_HOST = "prism.nacse.org"
_PRISM_FTP_DIR = "monthly"

# increment between successive lat and lon values
_LAT_LON_INCREMENT = 1 / 24.0

# ------------------------------------------------------------------------------
# set up a basic, global logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d  %H:%M:%S")
logger = logging.getLogger(__name__)


def month_range(start_dt, end_dt):
    import copy
    curr_dt = copy.copy(start_dt)
    while curr_dt <= end_dt:
        yield curr_dt
        curr_dt += relativedelta(months=1)


# ------------------------------------------------------------------------------
def download_files(
        dest_dir: str,
        year_month_initial: str,
        year_month_final: str,
        var_name: str,
) -> List[str]:
    """

    :param year_month_initial: starting year and month of date range (inclusive),
        with format "YYYYMM"
    :param year_month_final: ending year and month of date range (inclusive),
        with format "YYYYMM"
    :return: list of monthly files at the PRISM FTP location that correspond
        to the specified date range
    """

    # Start with a list of dates,
    #   then check if the bils exist, then check if the zips exist
    start_dt = datetime.strptime(f'{year_month_initial}01', '%Y%m%d')
    end_dt = datetime.strptime(f'{year_month_final}01', '%Y%m%d')
    provisional_dt = (datetime.today() - relativedelta(months=6))
    provisional_dt = datetime(provisional_dt.year, provisional_dt.month, 1)
    date_list = list(month_range(start_dt, end_dt))

    # Build the year folders if they don't exist
    for year in sorted(list(set(dt.year for dt in date_list))):
        year_ws = os.path.join(dest_dir, var_name, str(year))
        if not os.path.isdir(year_ws):
            os.makedirs(year_ws)

    zip_download_list = []
    bil_zip_dict = {}
    bil_list = []
    for tgt_dt in date_list:
        # TODO: Should probably get the list of provisional from the FTP instead of assuming
        if tgt_dt >= provisional_dt:
            # print(f'{tgt_dt} Provisional')
            # print(f'{provisional_dt}')
            zip_name = f'PRISM_{var_name}_provisional_4kmM3_{tgt_dt.strftime("%Y%m")}_bil.zip'
            bil_name = f'PRISM_{var_name}_provisional_4kmM3_{tgt_dt.strftime("%Y%m")}_bil.bil'
            # print(zip_file)
            # print(bil_file)
            # input('ENTER')
        elif (tgt_dt.year >= 1981) and (tgt_dt < provisional_dt):
            zip_name = f'PRISM_{var_name}_stable_4kmM3_{tgt_dt.strftime("%Y%m")}_bil.zip'
            bil_name = f'PRISM_{var_name}_stable_4kmM3_{tgt_dt.strftime("%Y%m")}_bil.bil'
        elif tgt_dt.year <= 1980:
            if var_name in ['ppt']:
                zip_name = f'PRISM_{var_name}_stable_4kmM2_{tgt_dt.year}_all_bil.zip'
                bil_name = f'PRISM_{var_name}_stable_4kmM2_{tgt_dt.strftime("%Y%m")}_bil.bil'
            else:
                zip_name = f'PRISM_{var_name}_stable_4kmM3_{tgt_dt.year}_all_bil.zip'
                bil_name = f'PRISM_{var_name}_stable_4kmM3_{tgt_dt.strftime("%Y%m")}_bil.bil'

        year_ws = os.path.join(dest_dir, var_name, str(tgt_dt.year))
        zip_path = os.path.join(year_ws, zip_name)
        bil_path = os.path.join(year_ws, bil_name)

        # TODO: Pull the full file list once instead of checking each file
        if os.path.isfile(bil_path):
            bil_list.append(bil_path)
            continue
        elif not os.path.isfile(bil_path) and os.path.isfile(zip_path):
            bil_zip_dict[bil_name] = zip_name
        else:
            zip_download_list.append(zip_name)
            bil_zip_dict[bil_name] = zip_name

    zip_download_list = sorted(list(set(zip_download_list)))
    # pprint.pprint(zip_list)
    # pprint.pprint(list(bil_zip_dict.keys()))
    # input('ENTER')

    # Download zip files
    if zip_download_list:
        with ftplib.FTP(host=_PRISM_FTP_HOST) as ftp:
            ftp.login("anonymous", "ftplib-example")
            for zip_name in zip_download_list:
                year = zip_name.split('_')[4][:4]
                zip_path = os.path.join(dest_dir, var_name, year, zip_name)
                ftp.cwd(f'/{_PRISM_FTP_DIR}/{var_name}/{year}')
                print(zip_name)
                ftp.retrbinary("RETR " + zip_name, open(zip_path, "wb").write)
            ftp.quit()

    # unzip the ZIP file and get the variable's bil file
    for bil_name, zip_name in bil_zip_dict.items():
        year = zip_name.split('_')[4][:4]
        year_ws = os.path.join(dest_dir, var_name, year)
        zip_path = os.path.join(year_ws, zip_name)
        with zipfile.ZipFile(zip_path) as zip_file:
            for file_name in zip_file.namelist():
                if file_name.split('.')[0] != bil_name.split('.')[0]:
                    continue
                elif not (file_name.endswith('.bil') or
                        file_name.endswith('.hdr') or
                        file_name.endswith('.prj')):
                    continue
                print(file_name)
                zip_file.extract(file_name, year_ws)

                if file_name.endswith('.bil'):
                    bil_list.append(bil_path)

    # try:
    #     os.remove(destination_path)
    # except:
    #     pass

    # TODO: Write a fancy one line sort
    return sorted([b for b in bil_list if 'stable' in b]) + \
           sorted([b for b in bil_list if 'provisional' in b])


# ------------------------------------------------------------------------------
# TODO extract this out into a netCDF utilities module, as it's
#  useful (and duplicated) in other codes
def compute_days(initial_year,
                 initial_month,
                 total_months,
                 start_year=1800):
    """
    Computes the "number of days" equivalent of the regular, incremental monthly
    time steps from an initial year/month.

    :param initial_year: the initial year from which the increments should start
    :param initial_month: the initial month from which the increments should start
    :param total_months: the total number of monthly increments (time steps
        measured in days) to be computed
    :param start_year: the start year from which the monthly increments
        (time steps measured in days) to be computed
    :return: an array of time step increments, measured in days since midnight
        of January 1st of the start_year
    :rtype: ndarray of ints
    """

    # compute an offset from which the day values should begin
    # (assuming that we're using "days since <start_date>" as time units)
    start_date = datetime(start_year, 1, 1)

    # initialize the list of day values we'll build
    days = np.empty(total_months, dtype=int)

    # loop over all time steps (months)
    for i in range(total_months):

        # the number of years since the initial year
        years = int((i + initial_month - 1) / 12)
        # the number of months since January
        months = int((i + initial_month - 1) % 12)

        # cook up a date for the current time step (month)
        current_date = datetime(initial_year + years, 1 + months, 1)

        # leverage the difference between dates operation available with datetime objects
        days[i] = (current_date - start_date).days

    return days


# ------------------------------------------------------------------------------
def get_coordinate_values(
        hdr_file: str,
) -> (np.ndarray, np.ndarray):
    """
    This function takes a PRISM BIL HDR file for a single month and extracts
    a list of lat and lon coordinate values for the regular grid contained therein.

    :param hdr_file:
    :return: lats and lons respectively
    :rtype: two 1-D numpy arrays of floats
    """

    with open(hdr_file) as hdr_f:
        hdr = {x.split()[0].strip(): x.split()[1].strip()
               for x in hdr_f.readlines() if x}
    # print(hdr)

    # TODO: Check if cellsize matches expected value
    # if int(hdr['YDIM']) != _LAT_LON_INCREMENT

    # The ULYMAP and ULXMAP values are cell centers
    # Build the lower left lat from the upper left value to match the nclimgrid tool
    lat_start = float(hdr["ULYMAP"]) - ((int(hdr['NROWS']) - 1) * _LAT_LON_INCREMENT)
    lat_values = (np.arange(int(hdr['NROWS'])) * _LAT_LON_INCREMENT) + lat_start
    lon_values = (np.arange(int(hdr['NCOLS'])) * _LAT_LON_INCREMENT) + \
                 float(hdr["ULXMAP"])

    return lat_values, lon_values


# ------------------------------------------------------------------------------
def get_variable_attributes(var_name):
    """
    This function builds a dictionary of variable attributes based on the
    variable name. Four variable names are supported: 'prcp', 'tave', 'tmin',
    and 'tmax'.

    :param var_name:
    :return: attributes relevant to the specified variable name
    :rtype: dictionary with string keys corresponding to attribute names
        specified by the NCEI NetCDF template for gridded datasets
        (see https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/grid.cdl)
    """

    # initialize the attributes dictionary with values
    # applicable to all supported variable names
    attributes = {
        "coordinates": "time lat lon",
        "references": "GHCN-Monthly Version 3 (Vose et al. 2011), NCEI/NOAA, https://www.ncdc.noaa.gov/ghcnm/v3.php",
    }

    # flesh out additional attributes, based on the variable type
    if var_name == "ppt":
        attributes["long_name"] = "Precipitation, monthly total"
        attributes["standard_name"] = "precipitation_amount"
        attributes["units"] = "millimeter"
        attributes["valid_min"] = np.float32(0.0)
        attributes["valid_max"] = np.float32(2000.0)
    else:
        attributes["standard_name"] = "air_temperature"
        attributes["units"] = "degree_Celsius"
        attributes["valid_min"] = np.float32(-100.0)
        attributes["valid_max"] = np.float32(100.0)
        if var_name == "tmean":
            attributes["long_name"] = "Temperature, monthly average of daily averages"
        elif var_name == "tmax":
            attributes["long_name"] = "Temperature, monthly average of daily maximums"
        elif var_name == "tmin":
            attributes["long_name"] = "Temperature, monthly average of daily minimums"
        else:
            raise ValueError(f"The variable_name argument \"{var_name}\" is unsupported.")

    return attributes


# ------------------------------------------------------------------------------
def initialize_dataset(
        template_hdr_file_path: str,
        var_name: str,
        year_start: int,
        month_start: int,
        year_end: int,
        month_end: int,
) -> xr.Dataset:
    """

    :param template_file_name:
    :param var_name:
    :param year_start:
    :param month_start:
    :param year_end:
    :param month_end:
    :return:
    """

    # determine the lat and lon coordinate values by extracting these
    # from the initial BIL file in our list (assumes that each BIL
    # file contains the same lat/lon coordinates)
    lat_values, lon_values = get_coordinate_values(template_hdr_file_path)

    min_lat = np.float32(min(lat_values))
    max_lat = np.float32(max(lat_values))
    min_lon = np.float32(min(lon_values))
    max_lon = np.float32(max(lon_values))
    lat_units = "degrees_north"
    lon_units = "degrees_east"
    total_lats = lat_values.shape[0]
    total_lons = lon_values.shape[0]

    # set global group attributes
    global_attributes = {
        "date_created": str(datetime.now()),
        "date_modified": str(datetime.now()),
        "Conventions": "CF-1.6, ACDD-1.3",
        "ncei_template_version": "NCEI_NetCDF_Grid_Template_v2.0",
        "title": "PRISM (monthly)",
        "naming_authority": "gov.noaa.ncei",
        "standard_name_vocabulary": "Standard Name Table v35",
        "institution": "National Centers for Environmental Information (NCEI), NOAA, Department of Commerce",
        "geospatial_lat_min": min_lat,
        "geospatial_lat_max": max_lat,
        "geospatial_lon_min": min_lon,
        "geospatial_lon_max": max_lon,
        "geospatial_lat_units": lat_units,
        "geospatial_lon_units": lon_units,
    }

    # create a time coordinate variable with one
    # increment per month of the period of record
    time_units_start_year = 1800
    total_months = ((year_end - year_start) * 12) + month_end - month_start + 1
    time_values = compute_days(year_start, month_start, total_months, time_units_start_year)
    time_attributes = {
        "long_name": "Time, in monthly increments",
        "standard_name": "time",
        "calendar": "gregorian",
        "units": f"days since {time_units_start_year}-01-01 00:00:00",
        "axis": "T",
    }
    time_variable = xr.Variable(dims="time", data=time_values, attrs=time_attributes)

    # create the lat coordinate variable
    lat_attributes = {
        "standard_name": "latitude",
        "long_name": "Latitude",
        "units": lat_units,
        "axis": "Y",
        "valid_min": min_lat,
        "valid_max": max_lat,
    }
    lat_variable = xr.Variable(dims="lat", data=lat_values, attrs=lat_attributes)

    # create the lon coordinate variable
    lon_attributes = {
        "standard_name": "longitude",
        "long_name": "Longitude",
        "units": lon_units,
        "axis": "X",
        "valid_min": min_lon,
        "valid_max": max_lon,
    }
    lon_variable = xr.Variable(dims="lon", data=lon_values, attrs=lon_attributes)

    # create the data variable's array
    variable_data = \
        np.full(
            (time_variable.shape[0], total_lats, total_lons),
            fill_value=np.float32(np.NaN),
            dtype=np.float32,
        )
    coordinates = {
        "time": time_variable,
        "lat": lat_variable,
        "lon": lon_variable,
    }
    data_array = \
        xr.DataArray(
            data=variable_data,
            coords=coordinates,
            dims=["time", "lat", "lon"],
            name=var_name,
            attrs=get_variable_attributes(var_name),
        )

    # package it all as an xarray.Dataset
    dataset = \
        xr.Dataset(
            data_vars={var_name: data_array},
            coords=coordinates,
            attrs=global_attributes,
        )

    return dataset


# ------------------------------------------------------------------------------
def read_bil(
        file_name: str,
) -> np.ndarray:
    """

    :param file_name:
    :return:
    """

    with rasterio.open(file_name) as src_ds:
        variable_data = src_ds.read(1)

    # Set nodata to NaN and flip
    variable_data[variable_data == -9999] = np.NaN
    variable_data = np.flipud(variable_data)

    return variable_data


# ------------------------------------------------------------------------------
def ingest_prism(
        var_name: str,
        dest_dir: str,
        date_start: str,
        date_end: str,
) -> str:
    """
    Ingests one of the four PRISM monthly variables into a NetCDF for a specified
    date range.

    :param var_name: name of variable, "ppt", "tmin", "tmax", or "tmean"
    :param dest_dir: directory where NetCDF file should be written
    :param date_start: starting year and month of date range (inclusive), with format "YYYYMM"
    :param date_end: ending year and month of date range (inclusive), with format "YYYYMM"
    :return:
    """

    logger.info(
        f"Ingesting PRISM monthly data for variable '{var_name}' "
        f"and date range {date_start} - {date_end}",
    )

    # download and extract the BIL files within our date range
    try:
        file_list = download_files(dest_dir, date_start, date_end, var_name)
    except Exception as ex:
        logger.error(f"Failed to get files for variable '{var_name}'", ex)
        raise ex

    # initialize the xarray.DataSet
    dataset = \
        initialize_dataset(
            file_list[0].replace('.bil', '.hdr'),
            var_name,
            int(date_start[:4]),
            int(date_start[4:]),
            int(date_end[:4]),
            int(date_end[4:]),
        )

    # for each month download the data from FTP and convert to a numpy array,
    # adding it into the xarray dataset at the appropriate index
    bil_files = [f for f in file_list if f.endswith('.bil')]
    for time_step, bil_file_name in enumerate(bil_files):

        # get the values for the month
        monthly_values = read_bil(bil_file_name)

        # add the monthly values into the data array for the variable
        dataset[var_name][time_step] = monthly_values

    # write the xarray DataSet as NetCDF file into the destination directory
    file_name = f"prism_{var_name}_{date_start}_{date_end}.nc"
    var_file_path = os.path.join(dest_dir, file_name)
    logger.info(f"Writing PRISM NetCDF {var_file_path}")
    dataset.to_netcdf(var_file_path)

    return var_file_path


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # parse the command line arguments
    argument_parser = argparse.ArgumentParser()
    # argument_parser.add_argument(
    #     "--dest_dir",
    #     required=True,
    #     help="directory where the final, full time series NetCDF files should be written",
    # )
    argument_parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="year/month date at which the dataset should start (inclusive), in 'YYYYMM' format",
    )
    argument_parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="year/month date at which the dataset should end (inclusive), in 'YYYYMM' format",
    )
    args = vars(argument_parser.parse_args())

    # create an iterable containing dictionaries of parameters, with one
    # dictionary of parameters per variable, since there will be a separate
    # ingest process per variable, with each process having its own set
    # of parameters
    variables = ["ppt", "tmean", "tmin", "tmax"]
    params_list = []
    for variable_name in variables:
        ingest_prism(
            variable_name,
            os.getcwd(),
            # args["dest_dir"],
            args["start"],
            args["end"],
        )

    # exit(0)

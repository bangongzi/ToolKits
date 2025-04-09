from osgeo import gdal
import math
import os
import sys
sys.path.append("../")
from utils.file_helper import makedir_if_not_exist, find_all_target_file
# from PIL import Image
# from collections.abc import Iterable
import numpy as np


def extract_subraster(input_raster, bounding_idx, output_path=None):
    """
    根据提供的边界索引从输入栅格中提取子区域，将其输出到内存或硬盘中。

    Args:
        input_raster (str or gdal.Dataset): 输入栅格，可以是 tif 文件路径或 GDAL Dataset 对象。
        bounding_idx (tuple of int): 包含四个整数的元组(min_col, min_row, col_size, row_size)，描述要提取的子区域。
        output_path (str, optional): 输出的文件路径。如果为 None，则返回内存中的 GDAL Dataset。

    Returns:
        gdal.Dataset or None: 如果 output_path 为空值，则函数返回提取后的子区域（基于内存存储的 GDAL Dataset）；
                              如果不为空，将子区域存储到 output_path 并返回 None。
    """
    # 检查是否提供了有效的边界索引
    if not isinstance(bounding_idx, tuple) or len(bounding_idx) != 4:
        raise ValueError("边界索引必须是一个包含四个整数的元组：(min_col, min_row, col_size, row_size)")

    # 打开栅格数据集
    if isinstance(input_raster, str):
        ds = gdal.Open(input_raster)
        if ds is None:
            raise IOError(f"无法打开栅格文件: {input_raster}")
    elif isinstance(input_raster, gdal.Dataset):
        ds = input_raster
    else:
        raise TypeError("input_raster 必须是字符串（文件路径）或 gdal.Dataset 对象")

    try:
        min_col, min_row, col_size, row_size = bounding_idx

        # 创建一个内存中的临时数据集用于保存剪裁结果
        mem_driver = gdal.GetDriverByName('MEM')
        mem_ds = mem_driver.Create('', col_size, row_size, ds.RasterCount, ds.GetRasterBand(1).DataType)

        # 设置新数据集的地理变换参数
        geotransform = ds.GetGeoTransform()
        new_geotransform = (
            geotransform[0] + min_col * geotransform[1] + min_row * geotransform[2],
            geotransform[1],
            geotransform[2],
            geotransform[3] + min_col * geotransform[4] + min_row * geotransform[5],
            geotransform[4],
            geotransform[5]
        )
        mem_ds.SetGeoTransform(new_geotransform)

        # 设置投影信息
        mem_ds.SetProjection(ds.GetProjection())

        # 进行剪裁操作
        for i in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(i)
            data = band.ReadAsArray(min_col, min_row, col_size, row_size)
            out_band = mem_ds.GetRasterBand(i)
            out_band.WriteArray(data)

        if output_path:
            output_path = os.path.join(output_path, 'test.tiff')
            # 将内存中的数据集写入磁盘文件
            out_driver = gdal.GetDriverByName('GTiff')
            if out_driver is None:
                raise RuntimeError("GTiff driver is not available.")
            out_ds = out_driver.CreateCopy(output_path, mem_ds, 0)
            out_ds.FlushCache()
            out_ds = None  # 关闭数据集以确保文件被正确写入磁盘
            # mem_ds = None

        return mem_ds
    finally:
        # 确保关闭原始数据集
        if not isinstance(input_raster, gdal.Dataset):
            ds = None


# 定义函数来获取tif图的四个角坐标
def get_corner_coordinates(tif_file):
    ds = gdal.Open(tif_file)
    gt = ds.GetGeoTransform()
    x_min = gt[0]
    y_max = gt[3]
    x_max = gt[0] + ds.RasterXSize * gt[1]
    y_min = gt[3] + ds.RasterYSize * gt[5]
    return x_min, y_max, x_max, y_min


def crop_overlapping_tiffs(tiff_paths, output_dir):
    """将输入的多个tiff文件overlap的区域切出来，放到硬盘上，供后续进一步处理。
    注意，这里的overlap区域依据投影坐标位置来进行选择。切分出来的各个文件栅格行数和列数不一定一致（像素大小可能不同）
    @tiff_paths: 输入的tiff文件路径列表
    @output_dir: 切分出来的重叠区域存放位置
    @return: list of str. crop后的tiff路径列表"""
    # 获取所有tif图的四个角坐标
    corner_coordinates = []
    for tif_file in tiff_paths:
        corner_coordinates.append(get_corner_coordinates(tif_file))

    # 计算所有tif图四角坐标的重合部分
    x_min = max([coord[0] for coord in corner_coordinates])
    y_max = min([coord[1] for coord in corner_coordinates])
    x_max = min([coord[2] for coord in corner_coordinates])
    y_min = max([coord[3] for coord in corner_coordinates])

    cropped_tiff_paths = []
    for tif_file in tiff_paths:
        ds = gdal.Open(tif_file)
        bands = ds.RasterCount
        gt = ds.GetGeoTransform()
        x_offset = int((x_min - gt[0]) / gt[1])
        y_offset = int((y_max - gt[3]) / gt[5])
        x_size = int((x_max - x_min) / gt[1])
        y_size = int((y_min - y_max) / gt[5])
    
        data = ds.ReadAsArray(x_offset, y_offset, x_size, y_size)
    
        if len(data.shape) < 3:
            data = np.expand_dims(data, axis=0)  # 将二维数组扩展为三维数组，表示只有一个波段
            
        os.makedirs(output_dir, exist_ok=True)
        driver = gdal.GetDriverByName('GTiff')
        cropped_tiff_path = os.path.join(output_dir, os.path.basename(tif_file))
        out_ds = driver.Create(cropped_tiff_path, x_size, y_size, bands,
                               ds.GetRasterBand(1).DataType)
    
        out_ds.SetGeoTransform((x_min, gt[1], 0, y_max, 0, gt[5]))
        out_ds.SetProjection(ds.GetProjection())
    
        for band in range(1, bands + 1):
            band_data = data[band - 1].astype(float)
            # band_data[band_data == 0] = np.nan
            out_ds.GetRasterBand(band).WriteArray(band_data)
        cropped_tiff_paths.append(cropped_tiff_path)
    return cropped_tiff_paths


def merge_tiff_files(tiff_paths, output_dir, data_type):
    """以增加图像通道数的方法合并多张tiff图为一张，并输出到硬盘上
    注意，此处合并的约束只是输入的多张tiff的行数和列数一致。如有其它合并图像约束，请在调用该函数前进行检验
    @tiff_paths: 待合并的tiff文件路径列表
    @output_dir: 合并后的tiff文件输出目录
    @data_type: 合并后的tiff文件数据存储类型，该类型应该能兼容输入的所有tiff文件的数据类型
    @return: str. 合并后的tiff文件路径"""
    x_size, y_size = None, None
    total_bands = 0
    for tiff_path in tiff_paths:
        ds = gdal.Open(tiff_path)
        if x_size:
            if x_size != ds.RasterXSize:
                raise RuntimeError(f"tiff files have different x_size: {x_size} vs {ds.RasterXSize}")
        else:
            x_size = ds.RasterXSize
        
        if y_size:
            if y_size != ds.RasterYSize:
                raise RuntimeError(f"tiff files have different y_size: {y_size} vs {ds.RasterYSize}")
        else:
            y_size = ds.RasterYSize
        
        total_bands += ds.RasterCount
    
    input_ds = gdal.Open(tiff_paths[0])
    driver = gdal.GetDriverByName('GTiff')

    output_path = os.path.join(output_dir, os.path.basename(tiff_paths[0]))
    out_ds = driver.Create(output_path, x_size, y_size, total_bands, data_type)
    out_ds.SetGeoTransform(input_ds.GetGeoTransform())  # 设置投影坐标等信息
    out_ds.SetProjection(input_ds.GetProjection())

    band_index = 0
    for tiff_path in tiff_paths:
        ds = gdal.Open(tiff_path)
        data = ds.ReadAsArray()
        if len(data.shape) < 3:
            data = np.expand_dims(data, axis=0)  # 只有一个波段时，将二维数组扩展为三维数组；大于一个波段时data自然时三维数组

        for j in range(1, ds.RasterCount + 1):
            band_data = data[j - 1].astype(np.float32)
            # band_data[band_data == 0] = np.nan
            band_index += 1
            out_ds.GetRasterBand(band_index).WriteArray(band_data)
    return output_path


def batch_crop_tiff_to_target_size(input_tiff_dir, target_size, cover_size, output_dir, edge_keep_ratio=0.05):
    """对一个文件夹下的所有tiff文件执行切分操作"""
    count = 0
    for tiff_dir, tiff_name in find_all_target_file(input_tiff_dir, (".tiff", )):
        tiff_path = os.path.join(tiff_dir, tiff_name)
        crop_tiff_to_target_size(tiff_path, target_size, cover_size, output_dir, edge_keep_ratio)
        count += 1
        if count % 10:
            print(f"{count} tiff files cropped for now")

    print(f"Successfully crop {count} tiff files for {input_tiff_dir}")


def crop_tiff_to_target_size(input_tiff_path, target_size, cover_size, output_dir, edge_keep_ratio=0.05):
    """切分大像素文件为更小（目标）像素文件
    @input_tiff_path: 输入的tiff文件路径
    @target_size: 目标文件（默认正方形）大小
    @cover: 重叠区域大小
    # todo 探讨切分到边缘时候怎么划分比较合适
    @edge_keep_ratio: 边缘保留门限值。
                      当边缘很小的时候（小于edge_keep_ratio * target_size），放弃切分该边缘值进样本，以免某些边缘区域出现重复太多，拉偏分布。
                      当边缘较大而不够target_size的时候，通过增大重叠区域面积的方式来使切分到边缘的区域达到target_size大小
    @output_dir: 目标输出路径
    @return: int. 切分出的文件数量"""
    os.makedirs(output_dir, exist_ok=True)
    ds = gdal.Open(input_tiff_path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    num_bands = ds.RasterCount
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    rows = math.ceil(height / (target_size - cover_size / 2))  # 向上取整
    cols = math.ceil(width / (target_size - cover_size / 2))

    file_count = 0
    for i in range(rows):
        for j in range(cols):
            xoff = j * (target_size - cover_size)
            yoff = i * (target_size - cover_size)
            if width - xoff < edge_keep_ratio * target_size:
                continue
            elif width - xoff < target_size:
                xoff = width - target_size

            if height - yoff < edge_keep_ratio * target_size:
                continue
            elif height - yoff < target_size:
                yoff = height - target_size

            data = []
            for band in range(1, num_bands + 1):
                band_data = ds.GetRasterBand(band).ReadAsArray(xoff, yoff, target_size, target_size)
                data.append(band_data)

            out_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_tiff_path))[0]}_{i}_{j}.tif")
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(out_path, target_size, target_size, num_bands, ds.GetRasterBand(1).DataType)
            out_ds.SetGeoTransform((gt[0] + xoff * gt[1], gt[1], 0, gt[3] + yoff * gt[5], 0, gt[5]))
            out_ds.SetProjection(proj)
            for band_idx in range(num_bands):
                out_ds.GetRasterBand(band_idx + 1).WriteArray(data[band_idx])
            file_count += 1


def batch_remove_tiff_white_border(input_file_dir, output_file_dir, threshold=(255, 0)):
    """
    批量去（带投影坐标系）tiff图像白边
    @input_file_dir 输入文件目录
    @output_file_dir 输出文件目录
    @threshold 阈值，用于判断是否为边框。标量或者iterable对象。
    """
    makedir_if_not_exist(output_file_dir)
    for idx, file_name in enumerate(os.listdir(input_file_dir)):
        input_file_path = os.path.join(input_file_dir, file_name)
        output_file_path = os.path.join(output_file_dir, file_name)
        remove_tiff_white_border(input_file_path, output_file_path, threshold)

        if (idx + 1) % 5 == 0:
            print(f"{idx + 1} files have been cleaned to {output_file_dir}")

    print(f"Removing white boarder finished for {input_file_dir}")
    

def remove_tiff_white_border(source_path, target_path, threshold=(255, 0)):
    """
    使用GDAL裁剪TIFF文件的白色边缘，并保持地理坐标系统不变。
    :param source_path: 输入TIFF文件路径。
    :param target_path: 输出裁剪后的TIFF文件路径。
    :param threshold: 阈值，用于判断是否为边框。标量或者iterable对象。
    :return: None
    """
    ds = gdal.Open(source_path, gdal.GA_ReadOnly)

    # 获取图像的基本信息
    width = ds.RasterXSize  # 列数
    height = ds.RasterYSize  # 行数
    bands = ds.RasterCount
    driver = ds.GetDriver()
    transform = ds.GetGeoTransform()
    proj = ds.GetProjection()

    def is_white(input_tuple, input_threshold):
        if not isinstance(input_threshold, Iterable):
            input_threshold = [input_threshold, ]

        for iter_t in input_threshold:
            if np.all(input_tuple == iter_t):
                return True

        return False

    # 一次性读取所有数据
    # 在选择边界时，提前一次性读取像素图层到data，相比于每个像素都调用ReadAsArray()接口，去边界耗时约减少80% （50秒耗时减少到13秒）
    # 读取的数据索引方式和ReadAsArray()的索引方式有差异（dim1和dim2互换），会增加代码理解成本。当前选择该执行速度快的代码实现方式。
    data = np.zeros((bands, height, width), dtype=np.uint8)
    for i in range(bands):
        band = ds.GetRasterBand(i + 1)
        data[i] = band.ReadAsArray()

    # 寻找顶部、底部、左侧和右侧的非白色边缘
    top, bottom, left, right = 0, 0, 0, 0

    # 寻找顶部
    while top < height and np.all([is_white(data[:3, top, x], threshold) for x in range(width)]):
        top += 1

    # 寻找底部
    while bottom < height and np.all(
            [is_white(data[:3, height - 1 - bottom, x], threshold) for x in range(width)]):
        bottom += 1

    # 寻找左侧
    while left < width and np.all([is_white(data[:3, y, left], threshold) for y in range(height)]):
        left += 1

    # 寻找右侧
    while right < width and np.all(
            [is_white(data[:3, y, width - 1 - right], threshold) for y in range(height)]):
        right += 1

    # 计算新的地理变换参数
    new_transform = list(transform)
    new_transform[0] += left * new_transform[1]
    # 注意，此处纵向的像素点大小为负值。因此该直接用加法来改变。
    new_transform[3] += top * new_transform[5]

    # 创建一个新的数据集以保存裁剪后的图像
    new_width = width - left - right
    new_height = height - top - bottom
    new_ds = driver.Create(target_path, new_width, new_height, bands, ds.GetRasterBand(1).DataType)
    new_ds.SetGeoTransform(new_transform)
    new_ds.SetProjection(proj)

    # 读取并写入裁剪后的图像数据
    for i in range(bands):
        band = ds.GetRasterBand(i + 1)
        data = band.ReadAsArray(left, top, new_width, new_height)
        new_band = new_ds.GetRasterBand(i + 1)
        new_band.WriteArray(data)


def remove_white_border(source_path, target_path, threshold=(255, 0)):
    """
    移除图像中前三个通道均同为threshold某个元素的像素边框。
    :param source_path: 输入PIL Image对象地址。
    :param target_path: 输出图像地址
    :param threshold: 阈值，用于判断是否为边框。标量或者iterable对象。
    :return: 移除边框后的图像。
    """
    # 将图像转换为可编辑的格式
    image = Image.open(source_path)
    pixels = image.load()

    # 获取图像尺寸
    width, height = image.size

    print([pixels[x, 4005][:4] for x in range(width)])
    print([pixels[x, 4004][:4] for x in range(width)])
    print([pixels[x, 0][:4] for x in range(width)])

    def is_white_boarder_pixel(input_tuple, input_threshold):
        if not isinstance(input_threshold, Iterable):
            input_threshold = [input_threshold, ]

        for iter_t in input_threshold:
            if input_tuple == (iter_t, iter_t, iter_t):
                return True

        return False

    # 计算需要移除的边框大小
    top, bottom, left, right = 0, 0, 0, 0
    for y in range(height):
        if not all(is_white_boarder_pixel(pixels[x, y][:3], threshold) for x in range(width)):
            break
        top += 1

    for y in range(height - 1, -1, -1):
        if not all(is_white_boarder_pixel(pixels[x, y][:3], threshold) for x in range(width)):
            break
        bottom += 1

    for x in range(width):
        if not all(is_white_boarder_pixel(pixels[x, y][:3], threshold) for y in range(height)):
            break
        left += 1

    for x in range(width - 1, -1, -1):
        if not all(is_white_boarder_pixel(pixels[x, y][:3], threshold) for y in range(height)):
            break
        right += 1

    # 切片移除边框
    image = image.crop((left, top, width - right, height - bottom))
    image.save(target_path)


def batch_resample_images(reference_file_path, input_file_dir, output_file_dir, reference_raster_size_only=True):
    """
    批量化重采样
     :param reference_file_path: 重采样参考文件路径
     :param input_file_dir: 输入路径
     :param output_file_dir: 输出路径
     :param reference_raster_size_only: 是否只采用参考文件的像素点长、宽作为采样依据
    """
    makedir_if_not_exist(output_file_dir)
    for idx, file_name in enumerate(os.listdir(input_file_dir)):
        input_file_path = os.path.join(input_file_dir, file_name)
        output_file_path = os.path.join(output_file_dir, file_name)
        resample_images(reference_file_path, input_file_path, output_file_path, reference_raster_size_only)

        if (idx + 1) % 50 == 0:
            print(f"{idx + 1} files have been resampled to {output_file_dir}")
    print(f"Resample finished for {input_file_dir}")


def resample_images(referencefilePath, inputfilePath, outputfilePath, reference_raster_size_only=True):  # 影像重采样
    """
    对单个文件进行重采样
     :param referencefilePath: 重采样参考文件路径
     :param inputfilePath: 输入路径
     :param outputfilePath: 输出路径
     :param reference_raster_size_only: 是否只采用参考文件的像素点长、宽作为采样依据
    """
    # 获取参考影像信息, 其实可以自定义这些信息，有参考的话就不用查这些参数了
    referencefile = gdal.Open(referencefilePath, gdal.GA_ReadOnly)
    referencefileProj = referencefile.GetProjection()
    referencefiletrans = referencefile.GetGeoTransform()
    bandreferencefile = referencefile.GetRasterBand(1)
    width = referencefile.RasterXSize
    height = referencefile.RasterYSize
    bands = referencefile.RasterCount
    # 获取输入影像信息
    inputrasfile = gdal.Open(inputfilePath, gdal.GA_ReadOnly)  # 打开输入影像
    inputProj = inputrasfile.GetProjection()  # 获取输入影像的坐标系
    if reference_raster_size_only:
        input_trans = inputrasfile.GetGeoTransform()
        width = math.ceil(inputrasfile.RasterXSize * input_trans[1] / referencefiletrans[1])
        height = math.ceil(inputrasfile.RasterYSize * input_trans[5] / referencefiletrans[5])
        bands = inputrasfile.RasterCount
        bandreferencefile = inputrasfile.GetRasterBand(1)
        # 除了像素点高、宽，trans中的其他属性直接使用输入遥感文件的属性
        referencefiletrans = (input_trans[0], referencefiletrans[1], input_trans[2],input_trans[3],input_trans[4], referencefiletrans[5])
        referencefileProj = inputProj

    # 创建重采样输出文件（设置投影及六参数）
    driver = gdal.GetDriverByName('GTiff')  # 这里需要定义，如果不定义自己运算会大大增加运算时间
    output = driver.Create(outputfilePath, width, height, bands, bandreferencefile.DataType)  # 创建重采样影像
    output.SetGeoTransform(referencefiletrans)  # 设置重采样影像的仿射矩阵为参考面的仿射矩阵
    output.SetProjection(referencefileProj)  # 设置重采样影像的坐标系为参考面的坐标系
    # 参数说明 输入数据集、输出文件、输入投影、参考投影、重采样方法(最邻近内插\双线性内插\三次卷积等)、回调函数
    # 针对遥感图像，常采用双线性插值来实现准度和计算量的平衡，参见：《南大洋海冰影像地图投影变换与瓦片切割应用研究》
    gdal.ReprojectImage(inputrasfile, output, inputProj, referencefileProj, gdal.GRA_Bilinear, 0.0, 0.0, )


if __name__ == "__main__":
    input_tiff_dir = r"/mnt/nvme1/Data/crm_zoom18_building_dataset/poi_distance_heatmap_uint8"
    target_size = 1024
    cover_size = 128
    output_dir = r"/mnt/nvme1/Data/crm_zoom18_building_dataset/poi_distance_heatmap_uint8_cropped_1024"
    batch_crop_tiff_to_target_size(input_tiff_dir, target_size, cover_size, output_dir)

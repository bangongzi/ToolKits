from osgeo import gdal, ogr
from osgeo.gdal import gdalconst
from functools import partial, cmp_to_key
# from data_process.utils.file_helper import load_yaml_file
import geopandas as gpd
import warnings
from rasterio.windows import Window
import rasterio
from shapely.geometry import box
from osgeo import gdal, ogr, osr
import os


def raster2vector(input_raster, band_idx=1, output_path=None, mask_raster=None, mask_band_idx=None):
    """
    将栅格数据转为矢量数据。

    Args:
        input_raster (str or gdal.Dataset): 输入的栅格数据集，可以是 tif 文件路径或 GDAL Dataset 对象。
        band_idx (int): 要转换的栅格文件通道，默认转换第一个通道。
        output_path (str, optional): 矢量输出目标地址。如果该地址为空，则直接返回内存中的矢量对象。
        mask_raster (str or gdal.Dataset, optional): 转化矢量时的掩码栅格数据集，可以是 tif 文件路径或 GDAL Dataset 对象。
                                                     如果是和 input_raster 一样的文件路径，则掩码通道和待转换通道是同一个。
        mask_band_idx (int, optional): 掩码使用的通道。如未指定，默认使用 band_idx。

    Returns:
        osgeo.ogr.DataSource or None: 转换后的矢量对象。如果 output_path 不为空，则此处返回空值。
    """
    # 打开栅格数据集
    if isinstance(input_raster, str):
        ds = gdal.Open(input_raster)
        if ds is None:
            raise IOError(f"无法打开栅格文件: {input_raster}")
        close_ds = True
    elif isinstance(input_raster, gdal.Dataset):
        ds = input_raster
        close_ds = False
    else:
        raise TypeError("input_raster 必须是字符串（文件路径）或 gdal.Dataset 对象")

    try:
        # 处理掩码栅格
        if mask_raster is not None:
            if isinstance(mask_raster, str) and mask_raster == input_raster:
                mask_ds = ds
                mask_band = mask_ds.GetRasterBand(band_idx if mask_band_idx is None else mask_band_idx)
            else:
                if isinstance(mask_raster, str):
                    mask_ds = gdal.Open(mask_raster)
                    if mask_ds is None:
                        raise IOError(f"无法打开掩码栅格文件: {mask_raster}")
                    close_mask_ds = True
                elif isinstance(mask_raster, gdal.Dataset):
                    mask_ds = mask_raster
                    close_mask_ds = False
                else:
                    raise TypeError("mask_raster 必须是字符串（文件路径）或 gdal.Dataset 对象")

                mask_band = mask_ds.GetRasterBand(band_idx if mask_band_idx is None else mask_band_idx)
        else:
            mask_band = None

        # 获取指定波段的数据
        band = ds.GetRasterBand(band_idx)

        # 创建一个内存中的矢量数据源
        drv = ogr.GetDriverByName('Memory') if output_path is None else ogr.GetDriverByName('ESRI Shapefile')
        if output_path:
            if os.path.exists(output_path):
                drv.DeleteDataSource(output_path)

        vector_ds = drv.CreateDataSource('out' if output_path is None else output_path)

        # 创建图层
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        layer = vector_ds.CreateLayer('polygonized', srs=srs)

        # 添加字段
        new_field = ogr.FieldDefn('DN', ogr.OFTInteger)
        layer.CreateField(new_field)

        # 将栅格数据转换为矢量数据
        gdal.Polygonize(band, mask_band, layer, 0, [], callback=None)
        """gdal.Polygonize() 做的事是将栅格图像中值相同的相邻栅格聚合为矢量多边形对象
            src_band 输入的栅格通道
            mask_band 掩码栅格通道，只有非0值对应的栅格才会被考虑生成矢量
            dst_layer 输出的矢量文件
            dst_attribute_index 聚合后的值存储在矢量文件的第几个属性
            options 选项
            callback 回调函数，用于在执行过程中报告进度等。
            这个函数接口并未找到官方文档介绍，因此有这里的详细注释"""

        # 如果指定了输出路径，则保存到磁盘并返回 None
        if output_path:
            return None
        else:
            return vector_ds
    finally:
        # 确保关闭原始数据集
        if close_ds:
            ds = None
        if 'mask_ds' in locals() and close_mask_ds:
            mask_ds = None

def extract_shp_file_in_target_region(shp_path, reference_raster_path, extra_margin=0):
    """提取目标区域内的矢量对象。注意，此处假设shp与raster文件的投影坐标系一致，如不一致，需要进行提前投影坐标系转换。
    @shp_path: 输入的矢量文件
    @reference_raster_patt: 目标区域参考文件
    @extra_margin: 在参考区域四周扩展的区域大小
    return: None"""
    building_polygons = gpd.read_file(shp_path)
    rs_ds = rasterio.open(reference_raster_path)
    transform = rs_ds.transform
    n_rows, n_cols = rs_ds.shape
    window = Window(col_off=0, row_off=0, width=n_cols, height=n_rows)
    window_bounds = rasterio.windows.bounds(window, transform)
    window_bounds = (window_bounds[0] - extra_margin,
                     window_bounds[1] - extra_margin,
                     window_bounds[2] + extra_margin,
                     window_bounds[3] + extra_margin)
    window_geom = box(*window_bounds)
    return building_polygons[building_polygons.intersects(window_geom)].copy()


def copy_shp_file(input_path, output_path):
    gdf = gpd.read_file(input_path)
    gdf.to_file(output_path, driver="ESRI Shapefile")


def map_vector_field(input_path, output_path, mapping_dict, allow_missing_key=True):
    # 读取输入，创建输出
    src_ds = ogr.Open(input_path, 0)
    driver = src_ds.GetDriver()
    dest_ds = driver.CreateDataSource(output_path)
    src_layer = src_ds.GetLayer()
    dest_layer = dest_ds.CreateLayer(src_layer.GetName(),
                                     src_layer.GetSpatialRef(),
                                     src_layer.GetGeomType())

    for i in range(src_layer.GetLayerDefn().GetFieldCount()):
        field_defn = src_layer.GetLayerDefn().GetFieldDefn(i)
        dest_layer.CreateField(field_defn)

    missing_key_dict = dict()
    for feature in src_layer:
        new_feature = ogr.Feature(dest_layer.GetLayerDefn())
        if feature.GetGeometryRef() is not None:
            new_feature.SetGeometry(feature.GetGeometryRef().Clone())
        else:
            warnings.warn(
                f"map_vector_field: {input_path} feature {feature.GetField('CARTO_ID')} has None GetGeometryRef {feature.GetGeometryRef()}")
            new_feature.SetGeometry(None)
        for i in range(feature.GetFieldCount()):
            new_feature.SetField(i, feature.GetField(i))

        for mapping_field_name, value_mapping_dict in mapping_dict.items():
            origin_value = feature.GetField(mapping_field_name)
            # 如果原来的值不是空值，加入这个转换逻辑；如果是空值，则赋空值（值复制已经在上边的循环执行过）
            if origin_value is not None:
                if origin_value in value_mapping_dict:
                    new_value = value_mapping_dict[origin_value]
                    new_feature.SetField(mapping_field_name, new_value)
                elif allow_missing_key:
                    if origin_value in missing_key_dict:
                        missing_key_dict[origin_value] += 1
                    else:
                        missing_key_dict[origin_value] = 1
                else:
                    raise ValueError(f"{origin_value} is not in mapping dict")

        dest_layer.CreateFeature(new_feature)
    if len(missing_key_dict) > 0:
        warnings.warn(f"missing mapping value {missing_key_dict} for {input_path}")
    dest_ds.FlushCache()

def vector2raster(inputfilePath, outputfile, templatefile, bands=[1], burn_values=[0], burn_field="", attribute_filter="", all_touch="False", data_type=gdal.GDT_Float32):
    # 打开矢量文件
    vector = ogr.Open(inputfilePath)
    # 获取矢量图层
    layer = vector.GetLayer()

    if attribute_filter:
        layer.SetAttributeFilter(attribute_filter)

    # 注意存储精度选择：gdal.GDT_Byte（8位无符号整数，0-255）, gdal.GDT_Float32, gdal.GDT_Float64。
    template_data = gdal.Open(templatefile, gdalconst.GA_ReadOnly)
    target_dataset = gdal.GetDriverByName('GTiff').Create(outputfile, template_data.RasterXSize, template_data.RasterYSize, 1, data_type)
    # 设置栅格坐标系与投影
    target_dataset.SetGeoTransform(template_data.GetGeoTransform())
    target_dataset.SetProjection(template_data.GetProjection())

    rasterize_options = ["ALL_TOUCHED=" + all_touch]
    if burn_field:
        rasterize_options.append("ATTRIBUTE=" + burn_field)
    # 调用栅格化函数。RasterizeLayer函数有四个参数，分别有栅格对象，波段，矢量对象，options
    # options可以有多个属性，其中ATTRIBUTE属性将矢量图层的某字段属性值作为转换后的栅格值
    gdal.RasterizeLayer(target_dataset, bands, layer, burn_values=burn_values,
                        options=rasterize_options)

    target_dataset.FlushCache()  # 确保数据写入磁盘


def str_float_field_cmp(feature1, feature2, field_name, desc=False):
    feature1 = feature1.GetField(field_name)
    feature2 = feature2.GetField(field_name)
    res = None
    # 认为None是最小的
    if not feature1 or not feature2:
        if feature2:
            res = -1
        elif feature1:
            res = 1
        else:
            res = 0
    else:
        feature1 = float(feature1)
        feature2 = float(feature2)

        if feature1 < feature2:
            res = -1
        elif feature1 > feature2:
            res = 1
        else:
            res = 0

    if desc:
        res = -res

    return res


def change_feature_order(input_path, output_path, customized_cmp_func):
    # 打开矢量文件
    src_ds = ogr.Open(input_path, 0)
    driver = src_ds.GetDriver()
    dest_ds = driver.CreateDataSource(output_path)
    src_layer = src_ds.GetLayer()
    dest_layer = dest_ds.CreateLayer(src_layer.GetName(),
                                     src_layer.GetSpatialRef(),
                                     src_layer.GetGeomType())

    for i in range(src_layer.GetLayerDefn().GetFieldCount()):
        field_defn = src_layer.GetLayerDefn().GetFieldDefn(i)
        dest_layer.CreateField(field_defn)

    src_feature_list = [feature for feature in src_layer]
    src_feature_list = sorted(src_feature_list, key=cmp_to_key(customized_cmp_func))

    for feature in src_feature_list:
        new_feature = ogr.Feature(dest_layer.GetLayerDefn())
        if feature.GetGeometryRef() is not None:
            new_feature.SetGeometry(feature.GetGeometryRef().Clone())
        else:
            warnings.warn(f"change_feature_order: {input_path} feature {feature.GetField('CARTO_ID')} has None GetGeometryRef {feature.GetGeometryRef()}")
            new_feature.SetGeometry(None)
        for i in range(feature.GetFieldCount()):
            new_feature.SetField(i, feature.GetField(i))

        dest_layer.CreateFeature(new_feature)
    dest_ds.FlushCache()


if __name__ == "__main__":
    inputfile_path = r'D:\Data\Exposure\05-SuperHighResolutionBuildingDataset\01-first-batch-exposure-sample\03-middle-layers\rasterized_buildings\debug\with_attribute\31.0_59.0_62.0_32.0_1.shp'
    output_shp_path = r'D:\Data\Exposure\05-SuperHighResolutionBuildingDataset\01-first-batch-exposure-sample\03-middle-layers\rasterized_buildings\debug\with_attribute\31.0_59.0_62.0_32.0_1_map_class_compact.shp'
    field_mapping_path = r"D:\Code\ExposureModeling\data_process\config\building_field_mapping_dict.yaml"

    # field_map = load_yaml_file(field_mapping_path)
    # map_vector_field(inputfile_path, output_shp_path, field_map)

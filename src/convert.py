import os
import shutil

import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    dataset_path = "/home/alex/DATASETS/TODO/QaTa-COV19/QaTa-COV19/QaTa-COV19-v2"

    batch_size = 30

    def create_ann_cg(image_path):
        tags = [cg_tag]

        img_height = 224
        img_wight = 224

        if cg2 is not None:
            image_np = sly.imaging.image.read(image_path)[:, :, 0]
            img_height = image_np.shape[0]
            img_wight = image_np.shape[1]

            tags.append(curr)

            if subfolder == "CHESTXRAY-14":
                tags.append(tr_test_tag)

        return sly.Annotation(img_size=(img_height, img_wight), img_tags=tags)

    def create_ann(image_path):
        labels = []

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        # img_height = image_np.shape[0]
        # img_wight = image_np.shape[1]

        mask_path = os.path.join(masks_path, "mask_" + get_file_name_with_ext(image_path))

        # if file_exists(mask_path):
        mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
        img_height = mask_np.shape[0]
        img_wight = mask_np.shape[1]
        mask = mask_np == 255
        ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
        for i in range(1, ret):
            obj_mask = curr_mask == i
            curr_bitmap = sly.Bitmap(obj_mask)
            if curr_bitmap.area > 100:
                curr_label = sly.Label(curr_bitmap, obj_class)
                labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    obj_class = sly.ObjClass("infected region", sly.Bitmap)

    group1_meta = sly.TagMeta("group 1", sly.TagValueType.NONE)
    group2_meta = sly.TagMeta("group 2", sly.TagValueType.NONE)
    chestxray_meta = sly.TagMeta("chestxray 14", sly.TagValueType.NONE)
    bacterial_meta = sly.TagMeta("pediatric bacterial pneumonia", sly.TagValueType.NONE)
    viral_meta = sly.TagMeta("pediatric viral pneumonia", sly.TagValueType.NONE)

    folder_to_meta = {
        "CHESTXRAY-14": chestxray_meta,
        "Pediatric_Bacterial_Pneumonia": bacterial_meta,
        "Pediatric_Viral_Pneumonia": viral_meta,
    }

    train_meta = sly.TagMeta("train", sly.TagValueType.NONE)
    test_meta = sly.TagMeta("test", sly.TagValueType.NONE)

    tr_test_to_meta = {"Train": train_meta, "Test": test_meta}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[obj_class],
        tag_metas=[
            group1_meta,
            group2_meta,
            chestxray_meta,
            bacterial_meta,
            viral_meta,
            train_meta,
            test_meta,
        ],
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name in os.listdir(dataset_path):

        images_path = os.path.join(dataset_path, ds_name, "Images")
        masks_path = os.path.join(dataset_path, ds_name, "Ground-truths")
        images_names = os.listdir(images_path)

        ds_name = ds_name.split(" ")[0].lower()

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(images_path, image_name) for image_name in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    # ===============================================

    ds_name = "control group"

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    cg2 = None

    images_path = "/home/alex/DATASETS/TODO/QaTa-COV19/QaTa-COV19/Control_Group/Control_Group_I"

    cg_tag = sly.Tag(group1_meta)

    images_names = os.listdir(images_path)

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

    for img_names_batch in sly.batched(images_names, batch_size=batch_size):
        images_pathes_batch = [
            os.path.join(images_path, image_name) for image_name in img_names_batch
        ]

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns_batch = [create_ann_cg(image_path) for image_path in images_pathes_batch]
        api.annotation.upload_anns(img_ids, anns_batch)

        progress.iters_done_report(len(img_names_batch))

    cg2 = True

    images_path = "/home/alex/DATASETS/TODO/QaTa-COV19/QaTa-COV19/Control_Group/Control_Group_II"

    cg_tag = sly.Tag(group2_meta)

    for subfolder in os.listdir(images_path):

        if subfolder != "CHESTXRAY-14":

            curr_meta = folder_to_meta[subfolder]
            curr = sly.Tag(curr_meta)

            curr_images_path = os.path.join(images_path, subfolder)

            images_names = os.listdir(curr_images_path)

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                images_pathes_batch = [
                    os.path.join(curr_images_path, image_name) for image_name in img_names_batch
                ]

                img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                anns_batch = [create_ann_cg(image_path) for image_path in images_pathes_batch]
                api.annotation.upload_anns(img_ids, anns_batch)

                progress.iters_done_report(len(img_names_batch))

        else:
            curr = sly.Tag(chestxray_meta)

            sub_images_path = os.path.join(images_path, subfolder)
            for tr_test in os.listdir(sub_images_path):

                tr_test_meta = tr_test_to_meta[tr_test]
                tr_test_tag = sly.Tag(tr_test_meta)

                curr_images_path = os.path.join(sub_images_path, tr_test)

                images_names = os.listdir(curr_images_path)

                progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

                for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                    images_pathes_batch = [
                        os.path.join(curr_images_path, image_name) for image_name in img_names_batch
                    ]

                    img_infos = api.image.upload_paths(
                        dataset.id, img_names_batch, images_pathes_batch
                    )
                    img_ids = [im_info.id for im_info in img_infos]

                    anns_batch = [create_ann_cg(image_path) for image_path in images_pathes_batch]
                    api.annotation.upload_anns(img_ids, anns_batch)

                    progress.iters_done_report(len(img_names_batch))

    return project

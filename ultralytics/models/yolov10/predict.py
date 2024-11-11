from ultralytics.models.yolo.detect import DetectionPredictor
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results
import logging

# Cấu hình logging
logging.basicConfig(
    filename='predict.log',  # Sử dụng đúng định dạng đường dẫn
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',  # Thêm định dạng cho log
    level=logging.INFO
)

logger = logging.getLogger()

class YOLOv10DetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        logger.info(f"Preds type: {type(preds)}, shape: {preds.shape if hasattr(preds, 'shape') else 'N/A'}")
        
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if preds.shape[-1] == 6:
            pass
        else:
           preds = preds.transpose(-1, -2)

            # Ghi thông tin về 'preds' sau khi hoán vị
           logger.info(f"'preds' sau khi hoán vị: shape = {preds.shape}, phần tử đầu tiên = {preds[0].tolist() if preds.numel() > 0 else 'N/A'}")

            # Thực hiện hàm v10postprocess để tính toán bboxes, scores, và labels
           bboxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, preds.shape[-1] - 4)

            # Ghi thông tin về 'bboxes', 'scores', và 'labels'
           logger.info(f"'bboxes': shape = {bboxes.shape}, phan tu dau tien = {bboxes[0].tolist() if bboxes.numel() > 0 else 'N/A'}")
           logger.info(f"'scores': shape = {scores.shape}, phan tu dau tien = {scores[0].tolist() if scores.numel() > 0 else 'N/A'}")
           logger.info(f"'labels': shape = {labels.shape}, phan tu dau tien = {labels[0].tolist() if labels.numel() > 0 else 'N/A'}")

            # Chuyển đổi định dạng bounding boxes từ xywh sang xyxy
           bboxes = ops.xywh2xyxy(bboxes)

            # Ghi thông tin về 'bboxes' sau khi chuyển đổi định dạng
           logger.info(f"'bboxes' sau khi chuyen doi xyxy: shape = {bboxes.shape}, phan tu dau tien = {bboxes[0].tolist() if bboxes.numel() > 0 else 'N/A'}")
            
           preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
        
        logger.info(f"Post-conversion preds shape: {preds.shape}")

        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
        
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]
        logger.info(f"Number of predictions after confidence and class filtering: {[len(p) for p in preds]}")

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            logger.info(f"Processing image {i + 1}/{len(orig_imgs)} with path: {self.batch[0][i]}")
            
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            
            # In thêm thông tin bboxes đã điều chỉnh
            logger.info(f"Image {i + 1}: Adjusted {len(pred)} bounding boxes to original image size: {pred[:, :4].tolist()}")

            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        
        logger.info("Finished processing all images, returning results.")
        return results


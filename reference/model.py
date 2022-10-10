import torchvision.models as models
import timm

def get_model(num_classes, pretrained=True):
	
	model = timm.create_model('swin_s3_base_224', pretrained=True,
                                   num_classes=num_classes)

	in_features = model.roi_heads.box_predictor.cls_score.in_features

	model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

	return model

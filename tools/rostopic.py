import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from cv_bridge import CvBridge
import cv2
import numpy as np

G1_INPUT_IMAGE_LEFT = "input/left/color/000012.jpg"
G1_INPUT_IMAGE_RIGHT = "input/right/color/000012.jpg"
G1_INPUT_IMAGE_INTRINSIC = "input/left/intrinsic/000012.txt" # intrinsic from left

G1_TOPIC_LEFT_COLOR = "/zed/zed_node/left/image_rect_color"
G1_TOPIC_RIGHT_COLOR = "/zed/zed_node/right/image_rect_color"
G1_TOPIC_CAMERA_INFO = "/zed/zed_node/left/camera_info" # intrinsic from left


class ZedCaptureNode(Node):
    def __init__(self, topic, file):
        self.topic = topic
        self.file = file
        topic_type = topic.split('/')[-1]
        topic_class = Image if topic_type == "image_rect_color" else CameraInfo
        super().__init__("zed_capture_node")
        self.subscription = self.create_subscription(
            topic_class,
            topic,
            self.zed_callback,
            10)
        self.bridge = CvBridge()
        self.received = False

    def zed_callback(self, msg):
        if not self.received:
            try:
                if self.topic.split('/')[-1] == "image_rect_color":
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    cv2.imwrite(self.file, cv_image)
                    self.received = True
                    self.get_logger().info(f"Zed image saved as {self.file}")
                elif self.topic.split('/')[-1] == "camera_info":
                    K = np.array(msg.k).reshape(3,3)
                    np.savetxt(self.file, K)
                    self.received = True
                    self.get_logger().info(f"Zed topic Message captured: {self.topic}")
                else:
                    # no matching zed topics found
                    self.received = True # prevent requesting for callback
                    self.get_logger().info(f"Unknown topic: {self.topic}")
                    
            except Exception as e:
                self.get_logger().error(f"Error capturing zed topics: {e}")

class PublishWeldpointsNode(Node):
    # get 2 x 3 numpy array for endpoints in the welding line obtained.
    def __init__(self, points: np.ndarray):
        super().__init__("weldpoint_publish_node")
        self.TOPIC_NAME = "eetrack_vision/weldpoints"
        assert points.shape == (2, 3), f"Array shape mismatch. Got {points.shape}. Expected (2, 3) shape array for welding line endpoints"
        self.points = points
        self.publisher = self.create_publisher(Float64MultiArray, self.TOPIC_NAME, 10)
        # create a timer
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.publish_weldpoints)

    def publish_weldpoints(self):
        # prepare msg data
        msg = Float64MultiArray()
        msg.data = list(self.points.flatten())

        row_dim = MultiArrayDimension()
        row_dim.label = "rows"
        row_dim.size = 2
        row_dim.stride = 3

        col_dim = MultiArrayDimension()
        col_dim.label = "columns"
        col_dim.size = 3
        col_dim.stride = 1

        msg.layout = MultiArrayLayout()
        msg.layout.dim = [row_dim, col_dim]
        msg.layout.data_offset = 0

        self.publisher.publish(msg)
        self.get_logger().info(f"Publishing {self.TOPIC_NAME}: {msg}") # just log msg.data for conciseness

def get_zed_images(args=None):
    rclpy.init(args=args)
    for topic, file, in zip([G1_TOPIC_LEFT_COLOR, G1_TOPIC_RIGHT_COLOR, G1_TOPIC_CAMERA_INFO], [G1_INPUT_IMAGE_LEFT, G1_INPUT_IMAGE_RIGHT, G1_INPUT_IMAGE_INTRINSIC]):
        node = ZedCaptureNode(topic, file)
        for _ in range(10):
            rclpy.spin_once(node)
        node.destroy_node()
    rclpy.shutdown()

def publish_weldpoints(points, args=None):
    rclpy.init(args=args)
    node = PublishWeldpointsNode(points)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    # default: get zed images and info (left, right, and intrinsic K)
    get_zed_images()
    # publish_weldpoints(np.random.random(size=(2,3)))

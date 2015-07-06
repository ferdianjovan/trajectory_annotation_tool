#!/usr/bin/env python

import cv2
import rospy
import Image
import Tkinter
import ImageTk
from cv_bridge import CvBridge
from std_msgs.msg import Header
from vision_people_logging.msg import LoggingUBD
from mongodb_store.message_store import MessageStoreProxy
from human_trajectory.trajectories import OfflineTrajectories
from human_trajectory_classifier.msg import TrajectoryType


class TRAN(object):

    def __init__(self, name):
        self.stored_uuids = list()
        self._uncertain_uuids = list()
        self.have_stored_uuids = list()
        self.trajectories = OfflineTrajectories()
        self.upbods = MessageStoreProxy(collection='upper_bodies').query(
            LoggingUBD._type, sort_query=[("$natural", 1)]
        )
        self._store_client = MessageStoreProxy(collection="trajectory_types")
        self.bridge = CvBridge()
        self.window = None
        self._traj_type = -1

    # create window for the image to show, and the button to click
    def _create_window(self, img):
        # main window
        self.window = Tkinter.Tk()
        self.window.title("Trajectory Annotator")

        # image frame in the top part of the main window
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        Tkinter.LabelFrame(self.window).pack()
        Tkinter.Label(self.window, image=imgtk).pack()

        # button frame in the bottom part of the main window
        buttons_frame = Tkinter.LabelFrame(self.window).pack(side=Tkinter.BOTTOM, expand="yes")
        # human
        human_button = Tkinter.Button(
            buttons_frame, text="Human", fg="black", command=self._human_button_cb
        )
        human_button.pack(side=Tkinter.LEFT)
        # non-human
        human_button = Tkinter.Button(
            buttons_frame, text="Non-Human", fg="black", command=self._nonhuman_button_cb
        )
        human_button.pack(side=Tkinter.RIGHT)

        # window main loop
        self.window.mainloop()

    # call back for human button
    def _human_button_cb(self):
        self.window.destroy()
        self._traj_type = 1

    # call back for non-human button
    def _nonhuman_button_cb(self):
        self.window.destroy()
        self._traj_type = 0

    # check in database whether the uuid exists
    def _check_mongo_for_uuid(self, uuids):
        for uuid in uuids:
            logs = self._store_client.query(
                TrajectoryType._type, message_query={"uuid": uuid}
            )
            if len(logs) and uuid not in self.have_stored_uuids:
                self.have_stored_uuids.append(uuid)

    # hidden annotation function for presenting image and get the vote
    def _annotate(self, ubd, index, uuids, annotation):
        stop = False
        self._check_mongo_for_uuid(uuids)

        if len(uuids) > 0 and not (set(uuids).issubset(self.stored_uuids) or set(uuids).issubset(self.have_stored_uuids)):
            if len(ubd[0].ubd_rgb) == len(ubd[0].ubd_pos):
                b, g, r = cv2.split(
                    self.bridge.imgmsg_to_cv2(ubd[0].ubd_rgb[index])
                )
                self._create_window(cv2.merge((r, g, b)))

                if int(self._traj_type) in [0, 1]:
                    for uuid in [i for i in uuids if i not in self.stored_uuids]:
                        rospy.loginfo("%s is stored..." % uuid)
                        annotation.update({uuid: int(self._traj_type)})
                        self.stored_uuids.append(uuid)
                        if len(uuids) > 1:
                            self._uncertain_uuids.append(uuid)

                    self._uncertain_uuids = [
                        i for i in self._uncertain_uuids if i not in uuids
                    ]
                else:
                    stop = True
            else:
                rospy.logwarn("UBD_RGB has %d data, but UBD_POS has %d data" % (len(ubd[0].ubd_rgb), len(ubd[0].ubd_pos)))
        # if all uuids have been classified before, then this removes
        # all doubts about those uuids
        elif set(uuids).issubset(self.stored_uuids):
            self._uncertain_uuids = [
                i for i in self._uncertain_uuids if i not in uuids
            ]

        return stop, annotation

    # annotation function
    def annotate(self):
        stop = False
        annotation = dict()
        for i in self.upbods:
            for j in range(len(i[0].ubd_pos)):
                uuids = self._find_traj_frm_pos(
                    i[0].header, i[0].ubd_pos[j], i[0].robot
                )
                stop, annotation = self._annotate(i, j, uuids, annotation)
                self._traj_type = -1
            if stop:
                break

        # storing the data
        counter = 1
        for uuid, value in annotation.iteritems():
            header = Header(counter, rospy.Time.now(), '')
            traj_type = 'human'
            if not value:
                traj_type = 'non-human'
            anno_msg = TrajectoryType(header, uuid, traj_type)
            if uuid in self.have_stored_uuids:
                self._store_client.update(
                    anno_msg, message_query='{"uuid":"%s"}' % uuid
                )
            else:
                self._store_client.insert(anno_msg)
            counter += 1

    # function to provide corresponding uuids based on time, human position, and
    # robot's position from UBD
    def _find_traj_frm_pos(self, header, point, robot):
        uuids = list()
        for uuid, traj in self.trajectories.traj.iteritems():
            stamps = [i[0].header.stamp for i in traj.humrobpose]
            index = self._header_index(header.stamp, stamps)
            points = [i[1].position for i in traj.humrobpose]
            index = self._point_index(robot.position, points, index)
            points = [i[0].pose.position for i in traj.humrobpose]
            index = self._point_index(point, points, index)
            if len(index) != 0:
                uuids.append(uuid.encode('ascii', 'ignore'))
        return uuids

    # function that returns indexes of time stamps whenever time stamp from UBD
    # matches time stamps from trajectories
    def _header_index(self, stamp, stamps):
        index = list()
        for i in range(len(stamps)):
            if stamps[0] > stamp:
                break
            if (stamps[i] - stamp).secs in [0, -1]:
                index.append(i)
        return index

    # function that returns indexes of human positions whenever the position
    # provided by UBD matches the positions from trajectories
    def _point_index(self, point, points, index=list()):
        index2 = list()
        dist = 0.1
        for i in index:
            if abs(point.x-points[i].x) < dist and abs(point.y-points[i].y) < dist:
                index2.append(i)
        return index2

if __name__ == '__main__':
    rospy.init_node("trajectory_annotation_node")
    tran = TRAN(rospy.get_name())
    tran.annotate()

from multiprocessing import Process

  # Start Initialization Class
    init=Initialization()
    # Open the app
    DB_Orig, DB_RealT, DB_Reset = init.app()
    # Database view
#    init.view_database(dir_image, dir_image_backup)


 
    # a custom function that blocks for a moment
    def task():
        # block for a moment
        init.view_database(dir_image, dir_image_backup)
    
    # entry point
    if __name__ == '__main__':
        # create a process
        process = Process(target=task)
        process.start()

############################
86 . self.name = name
99
        cv2.putText(frame_gui, 'Name: ' + str(self.name), 
                            (bbox.x1, bbox.y1-50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)


220
            bbox = BoundingBox(left*4,top*4,right*4,bottom*4)
            for tracker in trackers:
                tracker_bbox = tracker.bboxes[-1]
                iou = bbox.computeIOU(tracker_bbox)
                print('IOU( T' + str(tracker.id) + ' D' + str(name) + ' ) = ' + str(iou))
                cv2.rectangle(image_gui,(bbox.x1,bbox.y1),(bbox.x2, bbox.y2),(255,255,0),3)
                cv2.rectangle(image_gui,(tracker_bbox.x1,tracker_bbox.y1),(tracker_bbox.x2, tracker_bbox.y2),(255,255,255),3)
                if iou > iou_threshold: # Associate detection with tracker 
                    tracker_name = name


174
  tracker.draw(image_gui,tracker_name) 

  83

   tracker_name = None


import traceback, sys

class Notify:
    def notify(self, message, notification_type="warning", depth=0, force=False):
        """
        Usage:

        - Creates a class based notification message

        Requires:

        - `message`:
            - Type: str
            - What: The message to warn users with
            - Note: Messages with `{class_name}` and `{method_name}` in them are formatted appropriately

        Optional:

        - `notification_type`:
            - Type: str
            - What: The type of notification to send (warning, verbose or exception)
            - Default: "warning"
            - Note: 
                - "warning" prints a warning message
                - "verbose" prints a verbose message
                - "exception" raises an exception with the message
        - `depth`:
            - Type: int
            - What: The depth of the nth call below the top of the method stack
            - Note: Depth starts at 0 (indicating the current method in the stack)
            - Default: 0

        Notes:

        - If `self.show_warning_stack=False`, does not print the stack trace
        - If `self.show_warnings=False`, supresses all warnings

        """
        notification_types={
            "warning": "WARNING",
            "verbose": "",
            "exception": "EXCEPTION"
        }
        message=f"{self.__class__.__name__}.{sys._getframe(depth).f_back.f_code.co_name} {notification_types.get(notification_type, '')}: {message}"
        if notification_type=="exception":
            raise Exception(message)
        elif notification_type=="warning":
            if self.__dict__.get('notify_warnings',True) or force:
                if self.__dict__.get('notify_warning_stack',True):
                    traceback.print_stack(limit=10)
                print(message)
        elif notification_type=="verbose" or force:
            if self.__dict__.get('notify_verbose', False):
                print(message)
        else:
            raise Exception(f"Invalid notification type. Must be one of: {list(notification_types.keys())}")
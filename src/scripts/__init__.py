'''
import src.scripts.config as config
import src.scripts.funciones as funciones
import src.scripts.logger_config as logger_config
import src.scripts.project_config as project_config
import src.scripts.utils as utils
import src.scripts.setup_project as setup_project
from config import __dirPaths__


__all__ = ['config', 
           'funciones', 
           'logger_config', 
           'project_config',
           'utils', 
           'setup_project']

if __name__ == '__dirPaths__':
    __dirPaths__()
'''
from .config import *
from .funciones import *
from .logger_config import *
from .project_config import *
from .utils import *
from .setup_project import *
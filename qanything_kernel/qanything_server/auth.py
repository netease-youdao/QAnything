import time
from datetime import datetime
from sanic_jwt import Responses, exceptions, protected
from sanic import request
from sanic.response import json as sanic_json
import jwt
from qanything_kernel.core.local_doc_qa import LocalDocQA
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.utils.general_utils import *

__all__ = ["login", "add_user", "list_users", "get_user", "change_user", "change_password", "delete_user",
           "list_roles", "add_role", "change_role", "delete_role", "QAResponses"]

async def login(req: request, *args, **kwargs):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    passwd = safe_get(req, 'password')
    if user_id is None or passwd is None:
        raise exceptions.AuthenticationFailed("Missing username or password.")
    debug_logger.info("login %s", user_id)
    passwd_infos = local_doc_qa.mysql_client.get_passwd(user_id)
    if passwd_infos is not None:
        if len(passwd_infos) > 0:
            if passwd_infos[0] != passwd:
                raise exceptions.AuthenticationFailed("Invalid username or password")
            return { 'user_id': user_id }
        else:
            raise exceptions.AuthenticationFailed("Invalid username or password")
    else:
        raise exceptions.AuthenticationFailed("db error.")

@protected()
async def add_user(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("add user %s", user_id)
    user_type = safe_get(req, 'user_type')
    if user_type is None:
        user_type = 'user'
    group_id = safe_get(req, 'group_id')
    if group_id is None:
        group_id = 0
    user_exist = local_doc_qa.mysql_client.check_user_exist_(user_id)
    if user_exist:
        return sanic_json({"code": 2001, "msg": "fail, user {} already exist".format(user_id)})

    _, id = local_doc_qa.mysql_client.add_user(user_id, group_id, user_type)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    return sanic_json({"code": 200, "msg": "success create user {}".format(user_id),
                       "data": {"user_id": user_id, "id": id, "timestamp": timestamp}})

@protected()
async def change_user(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    debug_logger.info("change user %s", user_id)
    user_exist = local_doc_qa.mysql_client.check_user_exist_(user_id)
    if not user_exist:
        return sanic_json({"code": 2001, "msg": "fail, user {} not exist".format(user_id)})
    profile_pic = safe_get(req, 'profile_pic')
    password = safe_get(req, 'password')
    user_state = safe_get(req, 'state')
    telephone = safe_get(req, 'telephone')
    region = safe_get(req, 'region')
    wechat_id = safe_get(req, 'wechat_id')
    role_ids = safe_get(req, 'role_ids')
    local_doc_qa.mysql_client.change_user(user_id, password, profile_pic, user_state, telephone, region, wechat_id, role_ids)
    return sanic_json({"code": 200, "msg": "change user {} success".format(user_id),
                       })

@protected()
async def list_users(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_infos = local_doc_qa.mysql_client.get_user_list()
    datas = []
    for user in user_infos:
        datas.append({
            "id":user[0],
            "pid":user[1],
            "user_id":user[2],
            "user_type":user[3]
        })
    return sanic_json({ "code": 200, "msg": "success",
                        "data": datas
                       })

@protected()
async def get_user(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    debug_logger.info("get user %s", user_id)
    user_exist = local_doc_qa.mysql_client.check_user_exist_(user_id)
    if not user_exist:
        return sanic_json({"code": 2001, "msg": "fail, user {} not exist".format(user_id)})
    user_infos = local_doc_qa.mysql_client.get_user(user_id)
    if user_infos is not None and len(user_infos) > 0:
        user0 = user_infos[0]
        return sanic_json({"code": 200, "msg": "success",
                       "data":{
                           "telephone":user0[0],
                           "password":user0[1],
                           "state":user0[2],
                           "wechat_id":user0[3],
                           "role_ids":user0[4].split(','),
                           "profile_pic":user0[5]
                       }
                           })
    else:
        return sanic_json({"code": 2002, "msg": "failed".format(user_id),
                       })

@protected()
async def delete_user(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    debug_logger.info("delete user %s", user_id)
    user_exist = local_doc_qa.mysql_client.check_user_exist_(user_id)
    if not user_exist:
        return sanic_json({"code": 2001, "msg": "fail, user {} not exist".format(user_id)})
    local_doc_qa.mysql_client.delete_user(user_id)
    return sanic_json({"code": 200, "msg": "delete user {} success".format(user_id),
                       })

@protected()
async def change_password(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    debug_logger.info("change password %s", user_id)
    user_exist = local_doc_qa.mysql_client.check_user_exist(user_id)
    if not user_exist:
        return sanic_json({"code": 2003, "msg": "fail, user {} not exist".format(user_exist)})
    old_password = safe_get(req, 'old_password')
    if old_password is None or old_password == '':
        return sanic_json({"code": 2004, "msg": "missing old password"})
    passwd_infos = local_doc_qa.mysql_client.get_passwd(user_id)
    if passwd_infos is not None:
        if len(passwd_infos) > 0:
            if passwd_infos[0] != old_password:
                return sanic_json({"code": 2004, "msg": "fail, invalid old password"})
    password = safe_get(req, 'new_password')
    local_doc_qa.mysql_client.change_passwd(user_id, password)
    return sanic_json({"code": 200, "msg": "change user {} password success".format(user_id)})

@protected()
async def list_roles(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    role_infos = local_doc_qa.mysql_client.get_role_list()
    datas = []
    for role in role_infos:
        datas.append({
            "role_id":role[0],
            "role_name":role[1]
        })
    return sanic_json({ "code": 200, "msg": "success",
                        "data": datas
                       })
@protected()
async def add_role(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    role_id = safe_get(req, 'role_id')
    if role_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    debug_logger.info("add role %s", role_id)
    role_name = safe_get(req, 'role_name')
    if role_name is None or role_name == '':
        return sanic_json({"code": 2003, "msg": f'未提供角色名，请检查！'})
    role_exist = local_doc_qa.mysql_client.check_role_exist_(role_id)
    if role_exist:
        return sanic_json({"code": 2001, "msg": "fail, role {} already exist".format(role_id)})

    local_doc_qa.mysql_client.add_role(role_id, role_name)
    return sanic_json({"code": 200, "msg": "success create role {}".format(role_id)})

@protected()
async def change_role(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    role_id = safe_get(req, 'role_id')
    if role_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    debug_logger.info("add role %s", role_id)
    role_name = safe_get(req, 'role_name')
    if role_name is None or role_name == '':
        return sanic_json({"code": 2003, "msg": f'未提供角色名，请检查！'})
    role_exist = local_doc_qa.mysql_client.check_role_exist_(role_id)
    if not role_exist:
        return sanic_json({"code": 2001, "msg": "fail, role {} not exist".format(role_id)})

    local_doc_qa.mysql_client.change_role(role_id, role_name)
    return sanic_json({"code": 200, "msg": "success change role {}".format(role_id)})

@protected()
async def delete_role(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    role_id = safe_get(req, 'role_id')
    debug_logger.info("delete role %s", role_id)
    user_exist = local_doc_qa.mysql_client.check_role_exist_(role_id)
    if not user_exist:
        return sanic_json({"code": 2001, "msg": "fail, role {} not exist".format(role_id)})
    local_doc_qa.mysql_client.delete_role(role_id)
    return sanic_json({"code": 200, "msg": "delete role {} success".format(role_id),
                       })


class QAResponses(Responses):
    @staticmethod
    def extend_authenticate(request,
                            user=None,
                            access_token=None,
                            refresh_token=None):
        r = jwt.decode(access_token, options={"verify_signature": False})
        exp  = r['exp']
        exp_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(exp))
        return {
            "code":200,
            "msg":"success",
            "data":{
                "access_token":access_token,
                "refresh_token": refresh_token,
                "exp": exp_time
            }
        }

    @staticmethod
    def extend_retrieve_user(request, user=None, payload=None):
        return {}

    @staticmethod
    def extend_verify(request, user=None, payload=None):
        return {}

    @staticmethod
    def extend_refresh(request,
                       user=None,
                       access_token=None,
                       refresh_token=None,
                       purported_token=None,
                       payload=None):
        r = jwt.decode(refresh_token, options={"verify_signature": False})
        exp = r['exp']
        exp_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp))
        return {
            "code": 200,
            "msg":"success",
            "data":{
                "access_token": access_token,
                "exp": exp_time
            }
        }

    @staticmethod
    def exception_response(request, exception):
        reasons = (
            exception.args[0]
            if isinstance(exception.args[0], list)
            else [exception.args[0]]
        )
        return sanic_json(
            {   "code": exception.status_code,
                "msg": reasons,
                "exception": exception.__class__.__name__},
            status=exception.status_code,
        )
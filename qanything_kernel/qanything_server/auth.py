from sanic_jwt import Responses, exceptions, protected
from sanic import request
from sanic.response import json as sanic_json
from qanything_kernel.core.local_doc_qa import LocalDocQA
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.utils.general_utils import *

__all__ = ["login", "QAResponses"]

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


class QAResponses(Responses):
    @staticmethod
    def extend_authenticate(request,
                            user=None,
                            access_token=None,
                            refresh_token=None):
        return {
            "code":200,
            "msg":"success",
            "data":{
                "access_token":access_token
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
        return {
            "code": 200,
            "msg":"success",
            "data":{
                "refresh_token":access_token
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
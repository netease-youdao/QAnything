interface IResCheck {
  [type: string]: any;
}
export const ResCodeConfig = {
  noPerssion: {
    code: 403,
  },
  noLogin: {
    code: 401,
  },
  isSuccess: {
    code: 0,
  },
};
const checkResStatus: IResCheck = {} as IResCheck;

Object.keys(ResCodeConfig).forEach((type: string) => {
  checkResStatus[type] = (code: string | number = '') => {
    return Number(code) === ResCodeConfig[type].code;
  };
});
export default checkResStatus;

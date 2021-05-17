import AsyncStorage  from '@react-native-async-storage/async-storage';

const deviceStorage = {
  async saveKey(key, valueToSave) {
    try {
      await AsyncStorage.setItem(key, valueToSave);
      console.log("saved");
    } catch (error) {
      console.log('AsyncStorage Error: ' + error.message);
    }
  },

  async loadJWT() {
    try {
      AsyncStorage.getItem('id_token', (err, result) => {
        if(result){
          console.log("In JWT load", result)
          return result;
        }else{
          return null;
        }
      });
      // await AsyncStorage.getItem('id_token').then((value)=>{
      //   if (value !== null) {
      //     console.log("in load if", value);
      //     return value;
      //   // this.setState({
      //   //   jwt: value,
      //   //   loading: false
      //   // });
      // } else {
      //     console.log("in load else JWT");
      //     return null;
      //   // this.setState({
      //   //   loading: false
      //   // });
      // }
      // });
      
    } catch (error) {
      console.log('AsyncStorage Error: ' + error.message);
      return null;
    }
  },

  async deleteJWT() {
    try{
      await AsyncStorage.clear().then((data)=>{
        console.log("deleted");
      });
      
    //   .then(
    //     () => {
    //       this.setState({
    //         jwt: ''
    //       })
    //     }
    //   );
    } catch (error) {
      console.log('AsyncStorage Error: ' + error.message);
    }
  },

  async loadUser() {
    try {
      AsyncStorage.getItem('user_data', (err, result) => {
        if(result){
          console.log('in load user', result)
          return result;
        }else{
          return null;
        }
      });
      // if (value !== null) {
      //     console.log("in load user if", value);
      //     return value;
      //   // this.setState({
      //   //   jwt: value,
      //   //   loading: false
      //   // });
      // } else {
      //     console.log("in load user else");
      //     return null;
      //   // this.setState({
      //   //   loading: false
      //   // });
      // }
    } catch (error) {
      console.log('AsyncStorage Error: ' + error.message);
      return null;
    }
  },

  
};

export default deviceStorage;